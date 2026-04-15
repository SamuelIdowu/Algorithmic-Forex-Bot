"""
Agent Registry — Auto-discovers all BaseAgent subclasses in the agents/ package.

Agents are sorted by their `priority` attribute so the orchestrator always
runs them in the correct order (data → quant → sentiment → risk → logger).

To disable an agent without deleting it: set `enabled = False` on the class,
or pass its name to discover_agents(disabled=["sentiment"]).
"""
import importlib
import inspect
import pkgutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def discover_agents(
    disabled: Optional[list[str]] = None,
) -> list:
    """
    Auto-discover all BaseAgent subclasses in the agents/ package.

    Args:
        disabled (list[str], optional): Agent names to skip (e.g. ["sentiment"]).

    Returns:
        list[BaseAgent]: All enabled agent instances, sorted by priority.
    """
    # Import here to avoid circular imports at module load time
    from agents.base_agent import BaseAgent

    disabled = disabled or []
    agents = []

    # Iterate over all modules in the agents package
    package = importlib.import_module("agents")
    package_path = getattr(package, "__path__", [])

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        # Skip the base and registry modules themselves
        if module_name in ("base_agent", "registry"):
            continue

        try:
            module = importlib.import_module(f"agents.{module_name}")
        except Exception as exc:
            logger.error(f"Registry: failed to import agents.{module_name}: {exc}")
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseAgent)
                and obj is not BaseAgent
                and obj.__module__ == f"agents.{module_name}"  # avoid re-importing base
            ):
                instance = obj()
                # Respect class-level `enabled` flag and runtime disabled list
                if not instance.enabled:
                    logger.info(f"Registry: skipping disabled agent {instance.name}")
                    continue
                if instance.name.lower() in [d.lower() for d in disabled]:
                    logger.info(f"Registry: skipping agent {instance.name} (CLI disabled)")
                    continue
                agents.append(instance)
                logger.info(f"Registry: registered {instance}")

    # Sort by priority ascending so lower-priority numbers run first
    agents.sort(key=lambda a: a.priority)

    logger.info(f"Registry: {len(agents)} agents loaded: {[a.name for a in agents]}")
    return agents
