"""MCP server wrapper for the agent tool surface.

Exposes the same Layer 1-4 tools (or the no-graph parallel surface) that the
in-process agent loop uses, but over the Model Context Protocol so any
MCP-compatible client — Cursor, Claude Desktop, the MCP Inspector — can drive
a session interactively.

This is the optional/demo path described in spec §13. The headless
experimental loop (`scripts/run_e2_agent.py`) is unaffected; the underlying
tool functions are shared between both modes.

Run::

    python -m policy_doctor.vlm.proposals.agents.mcp_server

Configuration is via environment variables (see ``server.py``); Cursor passes
them through ``mcp.json``.
"""
