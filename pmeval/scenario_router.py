from pmeval.eval_template import list_templates


DEFAULT_TEMPLATE_ID = "product_recommendation"

SCENARIO_TEMPLATE_MAP = {
    "商品推荐": "product_recommendation",
    "妆容方案": "product_recommendation",
    "知识问答": "search",
    "搜索/评论综述": "search",
    "多轮对话": "customer_service",
    "边界/模糊问题": "customer_service",
    "内容推荐": "content_recommendation",
}


def known_template_ids() -> set[str]:
    return {template.id for template in list_templates()}


def resolve_template_id(scenario_type: str, template_id: str = "", default: str = DEFAULT_TEMPLATE_ID) -> str:
    explicit = str(template_id or "").strip()
    if explicit:
        return explicit
    return SCENARIO_TEMPLATE_MAP.get(str(scenario_type or "").strip(), default)


def is_mixed_template(template_ids: list[str]) -> bool:
    return len({item for item in template_ids if item}) > 1
