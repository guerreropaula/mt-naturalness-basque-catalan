"""Model-family-specific prompt rendering helpers."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

from src.utils.config import ModelEntry


_LANGUAGE_NAMES = {
    "en": "English",
    "ca": "Catalan",
    "val": "Valencian",
    "eu": "Basque",
}


def _language_name(language_code: str) -> str:
    return _LANGUAGE_NAMES.get(language_code, language_code)


def _prompt_context(
    source_text: str,
    target_lang: str,
    extra_template_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = {
        "source": source_text,
        "source_text": source_text,
        "english_source_text": source_text,
        "target_lang": target_lang,
        "target_lang_code": target_lang,
        "target_language_name": _language_name(target_lang),
        "target_language": _language_name(target_lang),
    }
    if extra_template_context:
        context.update(extra_template_context)
    return context


def _render_template(template: str, context: Mapping[str, Any]) -> str:
    try:
        return template.format(**context)
    except KeyError as exc:  # pragma: no cover - depends on config values
        missing_key = exc.args[0]
        raise ValueError(f"Prompt template referenced missing key '{missing_key}'") from exc


def _default_direct_translation_prompt(context: Mapping[str, Any]) -> str:
    return (
        f"Translate the following English text into {context['target_language_name']}. "
        "Return only the translation.\n\n"
        f"{context['source_text']}"
    )


def build_translation_messages(
    source_text: str,
    target_lang: str,
    prompt_spec: Mapping[str, Any] | None = None,
    extra_template_context: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build chat-style translation messages from a configurable prompt spec."""
    context = _prompt_context(
        source_text,
        target_lang,
        extra_template_context=extra_template_context,
    )
    if prompt_spec is None:
        return [{"role": "user", "content": _default_direct_translation_prompt(context)}]
    if prompt_spec.get("single_user_prompt_template") is not None:
        single_user_context = {**context, "target_lang": context["target_language_name"]}
        return [
            {
                "role": "user",
                "content": _render_template(
                    str(prompt_spec["single_user_prompt_template"]), single_user_context
                ),
            }
        ]

    default_system_prompt = (
        "You are a careful translation assistant. "
        "Translate the user input accurately into "
        f"{context['target_language_name']}. Return only the translation."
    )
    default_user_prompt = "{source_text}"

    system_template = (
        str(prompt_spec["system_prompt"])
        if prompt_spec and prompt_spec.get("system_prompt") is not None
        else default_system_prompt
    )
    user_template = (
        str(prompt_spec["user_prompt_template"])
        if prompt_spec and prompt_spec.get("user_prompt_template") is not None
        else default_user_prompt
    )
    return [
        {
            "role": "system",
            "content": _render_template(system_template, context),
        },
        {"role": "user", "content": _render_template(user_template, context)},
    ]


def _build_salamandrata_messages(context: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build the direct-MT user turn documented for SalamandraTA."""
    target_language = str(context["target_language_name"])
    return [
        {
            "role": "user",
            "content": (
                f"Translate the following text from English into {target_language}.\n"
                f"English: {context['source_text']}\n"
                f"{target_language}:"
            ),
        }
    ]


def _build_model_messages(
    model_entry: ModelEntry,
    source_text: str,
    target_lang: str,
    prompt_spec: Mapping[str, Any] | None,
    extra_template_context: Mapping[str, Any] | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    context = _prompt_context(
        source_text,
        target_lang,
        extra_template_context=extra_template_context,
    )
    if model_entry.prompt_adapter == "salamandrata_translation":
        prompt_label = None if prompt_spec is None else prompt_spec.get("label")
        if prompt_label not in {None, "p0_baseline", "p4_p5_direct_translation"}:
            raise ValueError(
                "SalamandraTA supports only its documented direct-translation prompt for P0/P4/P5; "
                f"received prompt variant {prompt_label!r}."
            )
        return _build_salamandrata_messages(context), context
    messages = build_translation_messages(
        source_text,
        target_lang,
        prompt_spec=prompt_spec,
        extra_template_context=extra_template_context,
    )
    if model_entry.family == "gemma3":
        # Gemma 3's processor expects multimodal-style content parts even for
        # text-only inference; this is the model-card chat-template schema.
        messages = [
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
            for message in messages
        ]
    return messages, context


def build_prompt_text(
    model_entry: ModelEntry,
    source_text: str,
    target_lang: str,
    tokenizer: Any | None = None,
    enable_thinking: bool | None = None,
    prompt_spec: Mapping[str, Any] | None = None,
    extra_template_context: Mapping[str, Any] | None = None,
) -> str:
    """Render a prompt using a tokenizer chat template when available."""
    messages, context = _build_model_messages(
        model_entry,
        source_text,
        target_lang,
        prompt_spec,
        extra_template_context,
    )
    chat_options = dict(model_entry.chat_template_options)

    effective_enable_thinking = (
        model_entry.enable_thinking if enable_thinking is None else enable_thinking
    )
    if prompt_spec and "enable_thinking" in prompt_spec and enable_thinking is None:
        effective_enable_thinking = bool(prompt_spec["enable_thinking"])
    if model_entry.family == "qwen3":
        chat_options["enable_thinking"] = bool(effective_enable_thinking)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        if model_entry.prompt_adapter == "salamandrata_translation":
            chat_options["date_string"] = date.today().isoformat()
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_options,
        )

    return messages[0]["content"]


def build_supervised_translation_text(
    model_entry: ModelEntry,
    source_text: str,
    target_text: str,
    target_lang: str,
    tokenizer: Any | None = None,
    enable_thinking: bool | None = None,
    prompt_spec: Mapping[str, Any] | None = None,
    extra_template_context: Mapping[str, Any] | None = None,
) -> str:
    """Render a full prompt+assistant training instance for token-length analysis."""
    messages, context = _build_model_messages(
        model_entry,
        source_text,
        target_lang,
        prompt_spec,
        extra_template_context,
    )
    chat_options = dict(model_entry.chat_template_options)

    effective_enable_thinking = (
        model_entry.enable_thinking if enable_thinking is None else enable_thinking
    )
    if prompt_spec and "enable_thinking" in prompt_spec and enable_thinking is None:
        effective_enable_thinking = bool(prompt_spec["enable_thinking"])
    if model_entry.family == "qwen3":
        chat_options["enable_thinking"] = bool(effective_enable_thinking)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        if model_entry.prompt_adapter == "salamandrata_translation":
            chat_options["date_string"] = date.today().isoformat()
        full_messages = [*messages, {"role": "assistant", "content": str(target_text)}]
        return tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **chat_options,
        )

    return build_prompt_text(
        model_entry,
        source_text=source_text,
        target_lang=target_lang,
        tokenizer=tokenizer,
        enable_thinking=enable_thinking,
        prompt_spec=prompt_spec,
        extra_template_context=extra_template_context,
    ) + str(target_text)
