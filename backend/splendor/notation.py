"""Move text notation and game-record (棋譜) serialization.

Text notation (human-readable, state-dependent on board slots):
    T:wug      take 3 different (white, blue, green)
    T:rr       take 2 of the same color (red)
    T:wu / T:r take fewer than 3 (only when the bank is short)
    R:L2#3     reserve the face-up Lv2 card in slot 3 (gains a gold)
    R:L1^      reserve the top card of the Lv1 deck (hidden)
    P:L3#1     purchase the face-up Lv3 card in slot 1
    P:H#2      purchase reserved card at hand index 2
Suffixes (appended with '/'):
    /D:rk      discard red + black down to the token cap
    /N:N3      visit noble N3

Canonical record format: ``seed`` + ordered ``moves`` fully reproduce every
position; analysis later annotates each ply with an ``eval`` object.
"""
from __future__ import annotations

import json

from .cards import ALL_TOKENS
from .state import GameState
from .moves import Move, apply_move, find_legal

ENGINE_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Move <-> notation
# ---------------------------------------------------------------------------
def to_notation(move: Move) -> str:
    if move.type == "take3":
        s = "T:" + "".join(move.gems)
    elif move.type == "take2":
        s = "T:" + move.color * 2
    elif move.type == "reserve":
        s = f"R:L{move.tier}^" if move.source == "deck" else f"R:L{move.tier}#{move.slot}"
    elif move.type == "purchase":
        s = f"P:H#{move.reserve_index}" if move.source == "reserve" else f"P:L{move.tier}#{move.slot}"
    else:
        raise ValueError(f"unknown move type: {move.type}")

    if move.discard:
        chars = "".join(c * move.discard[c] for c in ALL_TOKENS if move.discard.get(c))
        s += f"/D:{chars}"
    if move.noble:
        s += f"/N:{move.noble}"
    return s


def from_notation(state: GameState, text: str) -> Move:
    parts = text.split("/")
    head = parts[0]
    discard: dict = {}
    noble = None
    for seg in parts[1:]:
        tag, _, val = seg.partition(":")
        if tag == "D":
            for ch in val:
                discard[ch] = discard.get(ch, 0) + 1
        elif tag == "N":
            noble = val

    tag, _, val = head.partition(":")
    if tag == "T":
        if len(val) == 2 and val[0] == val[1]:
            m = Move(type="take2", color=val[0])
        else:
            m = Move(type="take3", gems=tuple(val))
    elif tag == "R":
        tier = int(val[1])
        if val.endswith("^"):
            cid = state.decks[tier][-1]
            m = Move(type="reserve", card_id=cid, source="deck", tier=tier)
        else:
            slot = int(val.split("#")[1])
            cid = state.board[tier][slot]
            m = Move(type="reserve", card_id=cid, source="board", tier=tier, slot=slot)
    elif tag == "P":
        if val.startswith("H#"):
            idx = int(val.split("#")[1])
            cid = state.current.reserved[idx]
            m = Move(type="purchase", card_id=cid, source="reserve", reserve_index=idx)
        else:
            tier = int(val[1])
            slot = int(val.split("#")[1])
            cid = state.board[tier][slot]
            m = Move(type="purchase", card_id=cid, source="board", tier=tier, slot=slot)
    else:
        raise ValueError(f"bad notation: {text}")

    m.discard = discard
    m.noble = noble
    return m


# ---------------------------------------------------------------------------
# Move <-> dict (full, state-independent action payload)
# ---------------------------------------------------------------------------
def move_to_dict(move: Move) -> dict:
    d = {"type": move.type}
    if move.type == "take3":
        d["gems"] = list(move.gems)
    elif move.type == "take2":
        d["color"] = move.color
    elif move.type == "reserve":
        d.update(cardId=move.card_id, source=move.source, tier=move.tier, slot=move.slot)
    elif move.type == "purchase":
        d.update(cardId=move.card_id, source=move.source, tier=move.tier,
                 slot=move.slot, reserveIndex=move.reserve_index)
    if move.discard:
        d["discard"] = dict(move.discard)
    if move.noble:
        d["noble"] = move.noble
    return d


def move_from_dict(d: dict) -> Move:
    return Move(
        type=d["type"],
        gems=tuple(d.get("gems", ())),
        color=d.get("color", ""),
        card_id=d.get("cardId", ""),
        source=d.get("source", ""),
        tier=d.get("tier", 0),
        slot=d.get("slot", -1),
        reserve_index=d.get("reserveIndex", -1),
        discard=dict(d.get("discard", {})),
        noble=d.get("noble"),
    )


# ---------------------------------------------------------------------------
# Game record (棋譜)
# ---------------------------------------------------------------------------
def new_record(player_specs: list[dict], seed: int, variant: str = "base") -> dict:
    return {
        "version": 1,
        "variant": variant,
        "players": [
            {"id": s["id"], "name": s.get("name", s["id"]), "type": s.get("type", "human"),
             "config": s.get("config", {})}
            for s in player_specs
        ],
        "seed": seed,
        "moves": [],
        "result": None,
        "meta": {"engineVersion": ENGINE_VERSION},
    }


def append_move(record: dict, state_before: GameState, move: Move) -> None:
    record["moves"].append({
        "ply": state_before.ply,
        "player": state_before.current.id,
        "notation": to_notation(move),
        "action": move_to_dict(move),
        "eval": None,
    })


def finalize_record(record: dict, state: GameState) -> dict:
    if state.is_over():
        record["result"] = {
            "winner": state.players[state.winner].id,
            "scores": [p.points for p in state.players],
            "plies": state.ply,
            "reason": "points>=15",
        }
    return record


def replay(record: dict, upto_ply: int | None = None) -> GameState:
    """Reconstruct the position after applying moves up to ``upto_ply`` (exclusive)."""
    state = GameState.new(record["players"], record["seed"])
    limit = len(record["moves"]) if upto_ply is None else upto_ply
    for entry in record["moves"][:limit]:
        move = move_from_dict(entry["action"])
        canonical = find_legal(state, move)
        if canonical is None:
            raise ValueError(f"illegal move during replay at ply {entry['ply']}: {entry['notation']}")
        apply_move(state, canonical)
    return state


def states_through(record: dict):
    """Yield (ply_index, state_before) for each move plus the final state."""
    state = GameState.new(record["players"], record["seed"])
    for i, entry in enumerate(record["moves"]):
        yield i, state.clone()
        move = find_legal(state, move_from_dict(entry["action"]))
        apply_move(state, move)
    yield len(record["moves"]), state


def save(record: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
