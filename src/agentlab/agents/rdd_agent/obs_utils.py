"""
Observation Utilities for RDD Planner

This module provides high-quality utility functions for extracting
goal text and accessibility tree information from BrowserGym observations.

These utilities are designed to be used with the RDD planner to provide
context about the current task and page state.
"""

from typing import Any, Optional
import json

from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str


def extract_goal_text(obs: dict[str, Any]) -> Optional[str]:
    """
    Extract natural-language goal text from a BrowserGym observation.
    
    This function attempts multiple strategies to find the goal text:
    1. Direct 'goal' field (string)
    2. First item in 'goal_object' tuple/list
    3. 'goal_object' dict with 'text' field
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Goal text as a string, or None if not found
    
    Examples:
        >>> obs = {"goal": "Sort the change requests list"}
        >>> extract_goal_text(obs)
        'Sort the change requests list'
        
        >>> obs = {"goal_object": ({"type": "text", "text": "Navigate to homepage"},)}
        >>> extract_goal_text(obs)
        'Navigate to homepage'
    """
    # Strategy 1: Direct goal field (most common for simple tasks)
    goal = obs.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal.strip()
    
    # Strategy 2: goal_object as tuple/list (multimodal format)
    goal_obj = obs.get("goal_object")
    if isinstance(goal_obj, (list, tuple)) and goal_obj:
        first = goal_obj[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            text = first["text"]
            if text.strip():
                return text.strip()
    
    # Strategy 3: goal_object as dict (alternative format)
    if isinstance(goal_obj, dict) and isinstance(goal_obj.get("text"), str):
        text = goal_obj["text"]
        if text.strip():
            return text.strip()
    
    # No goal found
    return None


def extract_axtree_flat(obs: dict[str, Any]) -> Optional[str]:
    """
    Extract flattened accessibility tree from a BrowserGym observation.
    
    This function handles two cases:
    1. Pre-flattened string in 'axtree_txt' field
    2. Raw AXTree object in 'axtree_object' that needs flattening
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Flattened accessibility tree as a string, or None if not available
    
    Examples:
        >>> obs = {"axtree_txt": "[1] RootWebArea 'Page Title'\\n  [2] button 'Click me'"}
        >>> extract_axtree_flat(obs)
        "[1] RootWebArea 'Page Title'\\n  [2] button 'Click me'"
        
        >>> obs = {"axtree_object": {"nodeId": 1, "role": "RootWebArea", ...}}
        >>> tree = extract_axtree_flat(obs)  # Returns flattened version
    """
    # Strategy 1: Already flattened string (most efficient)
    axtree_txt = obs.get("axtree_txt")
    if isinstance(axtree_txt, str) and axtree_txt.strip():
        return axtree_txt
    
    # Strategy 2: Raw object that needs flattening
    axtree_object = obs.get("axtree_object")
    if axtree_object is not None:
        try:
            flattened = flatten_axtree_to_str(axtree_object)
            if flattened and flattened.strip():
                return flattened
        except Exception as e:
            # Log error but don't crash
            import sys
            print(f"Warning: Failed to flatten AXTree: {e}", file=sys.stderr)
            return None
    
    # No accessibility tree found
    return None


def extract_axtree_object(obs: dict[str, Any]) -> Optional[dict]:
    """
    Extract raw accessibility tree object from a BrowserGym observation.
    
    This returns the structured AXTree object, not the flattened string.
    Useful for custom processing or analysis.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Raw AXTree object as a dictionary, or None if not available
    
    Examples:
        >>> obs = {"axtree_object": {"nodes": [{"nodeId": 1, ...}]}}
        >>> extract_axtree_object(obs)
        {"nodes": [{"nodeId": 1, ...}]}
    """
    axtree_object = obs.get("axtree_object")
    if axtree_object is not None and isinstance(axtree_object, dict):
        return axtree_object
    return None


def extract_html(obs: dict[str, Any]) -> Optional[str]:
    """
    Extract HTML content from a BrowserGym observation.
    
    This function attempts to extract HTML from various fields:
    1. 'dom_txt' field (pre-extracted HTML string)
    2. 'dom_object' field (structured DOM snapshot that needs flattening)
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        HTML content as a string, or None if not available
    
    Examples:
        >>> obs = {"dom_txt": "<html><body>Hello</body></html>"}
        >>> extract_html(obs)
        '<html><body>Hello</body></html>'
        
        >>> obs = {"dom_object": {"documents": [...], "strings": [...], ...}, "extra_element_properties": {...}}
        >>> html = extract_html(obs)  # Returns flattened HTML
    """
    # Strategy 1: Pre-extracted HTML string (most efficient)
    dom_txt = obs.get("dom_txt")
    if isinstance(dom_txt, str) and dom_txt.strip():
        return dom_txt
    
    # Strategy 2: Raw DOM object that needs flattening
    dom_object = obs.get("dom_object")
    if dom_object is not None:
        try:
            # Get extra element properties if available
            extra_properties = obs.get("extra_element_properties")
            
            # Flatten DOM object to HTML string
            if extra_properties is not None:
                flattened = flatten_dom_to_str(dom_object, extra_properties=extra_properties)
            else:
                flattened = flatten_dom_to_str(dom_object)
            
            if flattened and flattened.strip():
                return flattened
        except Exception as e:
            # Log error but don't crash
            import sys
            print(f"Warning: Failed to flatten DOM: {e}", file=sys.stderr)
            return None
    
    # No HTML found
    return None


def extract_dom_snapshot(obs: dict[str, Any]) -> Optional[dict]:
    """
    Extract raw DOM snapshot object from a BrowserGym observation.
    
    This returns the structured DOM object, not the HTML string.
    Useful for custom processing or analysis.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Raw DOM snapshot object as a dictionary, or None if not available
    
    Examples:
        >>> obs = {"dom_object": {"nodes": [{"nodeId": 1, ...}]}}
        >>> extract_dom_snapshot(obs)
        {"nodes": [{"nodeId": 1, ...}]}
    """
    dom_object = obs.get("dom_object")
    if dom_object is not None and isinstance(dom_object, dict):
        return dom_object
    return None


def extract_page_state(obs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract current page state information from a BrowserGym observation.
    
    This includes URL, page title, and focused element information.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Dictionary containing page state fields:
        - url: Current page URL
        - page_title: Title of the current page
        - focused_element_bid: Browser ID of the focused element
        - open_pages_urls: Tuple of all open page URLs
        - open_pages_titles: Tuple of all open page titles
        - active_page_index: Index of the currently active page
    
    Examples:
        >>> obs = {
        ...     "url": "https://example.com",
        ...     "open_pages_titles": ("Example Page",)
        ... }
        >>> state = extract_page_state(obs)
        >>> state["url"]
        'https://example.com'
    """
    return {
        "url": obs.get("url"),
        "page_title": obs.get("open_pages_titles", (None,))[0] if obs.get("open_pages_titles") else None,
        "focused_element_bid": obs.get("focused_element_bid"),
        "open_pages_urls": obs.get("open_pages_urls"),
        "open_pages_titles": obs.get("open_pages_titles"),
        "active_page_index": obs.get("active_page_index"),
    }


def extract_rdd_context(obs: dict[str, Any]) -> dict[str, Any]:
    """
    Extract all relevant context for RDD planning from a BrowserGym observation.
    
    This is a convenience function that combines goal text, accessibility tree,
    HTML content, and page state into a single JSON-serializable dictionary.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Dictionary containing:
        - goal_text: Natural-language task description
        - axtree_flat: Flattened accessibility tree string
        - html: HTML content of the page
        - page_state: Current page state information
        - has_goal: Boolean indicating if goal was found
        - has_axtree: Boolean indicating if accessibility tree was found
        - has_html: Boolean indicating if HTML was found
    
    Examples:
        >>> obs = {
        ...     "goal": "Sort the list",
        ...     "axtree_txt": "[1] RootWebArea",
        ...     "dom_txt": "<html>...</html>",
        ...     "url": "https://example.com"
        ... }
        >>> context = extract_rdd_context(obs)
        >>> context["goal_text"]
        'Sort the list'
        >>> context["has_goal"]
        True
    """
    goal_text = extract_goal_text(obs)
    axtree_flat = extract_axtree_flat(obs)
    html = extract_html(obs)
    page_state = extract_page_state(obs)
    
    return {
        "goal_text": goal_text,
        "axtree_flat": axtree_flat,
        "html": html,
        "page_state": page_state,
        "has_goal": goal_text is not None,
        "has_axtree": axtree_flat is not None,
        "has_html": html is not None,
    }


def format_state_description(obs: dict[str, Any]) -> str:
    """
    Format a human-readable state description for RDD planning.
    
    This creates a natural-language description of the current page state
    that can be used as the 'state' parameter for the RDD planner.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
    
    Returns:
        Human-readable state description string
    
    Examples:
        >>> obs = {
        ...     "url": "https://example.com/login",
        ...     "open_pages_titles": ("Login Page",)
        ... }
        >>> format_state_description(obs)
        'On page "Login Page" at https://example.com/login'
    """
    page_state = extract_page_state(obs)
    
    parts = []
    
    # Add page title if available
    if page_state["page_title"]:
        parts.append(f'On page "{page_state["page_title"]}"')
    
    # Add URL if available
    if page_state["url"]:
        if parts:
            parts.append(f'at {page_state["url"]}')
        else:
            parts.append(f'On page at {page_state["url"]}')
    
    # Fallback if no information available
    if not parts:
        return "On current page"
    
    return " ".join(parts)


def export_context_json(obs: dict[str, Any], pretty: bool = True) -> str:
    """
    Export RDD context as a JSON string.
    
    This is useful for debugging, logging, or saving context to a file.
    
    Args:
        obs: BrowserGym observation dictionary from env.reset() or env.step()
        pretty: If True, format JSON with indentation for readability
    
    Returns:
        JSON string containing all RDD context
    
    Examples:
        >>> obs = {"goal": "Test task", "url": "https://example.com"}
        >>> json_str = export_context_json(obs)
        >>> "goal_text" in json_str
        True
    """
    context = extract_rdd_context(obs)
    
    if pretty:
        return json.dumps(context, indent=2, ensure_ascii=False, default=str)
    else:
        return json.dumps(context, ensure_ascii=False, default=str)


__all__ = [
    "extract_goal_text",
    "extract_axtree_flat",
    "extract_axtree_object",
    "extract_html",
    "extract_dom_snapshot",
    "extract_page_state",
    "extract_rdd_context",
    "format_state_description",
    "export_context_json",
]
