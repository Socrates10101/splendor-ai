"""Game state: the single source of truth for a Splendor position.

``GameState`` is mutable and is advanced in place by :mod:`splendor.moves`.
``clone()`` makes a fast manual deep copy for tree search.

Hidden information: a card reserved from the *top of a deck* is visible only to
its owner. ``view_state(perspective)`` masks such cards for everyone else.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cards import CARDS, CARDS_BY_TIER, NOBLE_IDS, NOBLES, COLORS, GOLD, ALL_TOKENS, Card
from .rng import Rng
from . import rules


@dataclass
class PlayerState:
    id: str
    name: str = ""
    ptype: str = "human"                       # "human" | "ismcts" | ...
    tokens: dict = field(default_factory=lambda: {c: 0 for c in ALL_TOKENS})
    bonuses: dict = field(default_factory=lambda: {c: 0 for c in COLORS})
    cards: list = field(default_factory=list)  # owned development card IDs
    reserved: list = field(default_factory=list)        # reserved card IDs
    reserved_hidden: list = field(default_factory=list)  # parallel bool: reserved from deck
    nobles: list = field(default_factory=list)
    points: int = 0

    def total_tokens(self) -> int:
        return sum(self.tokens.values())

    def clone(self) -> "PlayerState":
        p = PlayerState(self.id, self.name, self.ptype)
        p.tokens = dict(self.tokens)
        p.bonuses = dict(self.bonuses)
        p.cards = list(self.cards)
        p.reserved = list(self.reserved)
        p.reserved_hidden = list(self.reserved_hidden)
        p.nobles = list(self.nobles)
        p.points = self.points
        return p


@dataclass
class GameState:
    num_players: int
    bank: dict                  # token color -> count in the supply
    decks: dict                 # tier -> list of card IDs (top = last element)
    board: dict                 # tier -> list of 4 slots (card ID or None)
    nobles: list                # available noble IDs
    players: list               # list[PlayerState]
    to_move: int = 0
    ply: int = 0
    winner: int | None = None   # player index, set once game over

    # ---- construction --------------------------------------------------
    @staticmethod
    def new(player_specs: list[dict], seed: int) -> "GameState":
        """player_specs: list of {id, name?, type?}."""
        n = len(player_specs)
        if n not in rules.TOKENS_BY_PLAYERS:
            raise ValueError(f"unsupported player count: {n}")
        rng = Rng(seed)

        conf = rules.TOKENS_BY_PLAYERS[n]
        bank = {c: conf["gems"] for c in COLORS}
        bank[GOLD] = conf["gold"]

        decks = {}
        board = {}
        for tier in (1, 2, 3):
            deck = list(CARDS_BY_TIER[tier])
            rng.shuffle(deck)
            slots = [deck.pop() if deck else None for _ in range(rules.BOARD_SLOTS_PER_TIER)]
            decks[tier] = deck
            board[tier] = slots

        noble_pool = list(NOBLE_IDS)
        rng.shuffle(noble_pool)
        nobles = noble_pool[: rules.nobles_in_play(n)]

        players = [
            PlayerState(
                id=spec["id"],
                name=spec.get("name", spec["id"]),
                ptype=spec.get("type", "human"),
            )
            for spec in player_specs
        ]
        return GameState(num_players=n, bank=bank, decks=decks, board=board,
                         nobles=nobles, players=players)

    # ---- cloning -------------------------------------------------------
    def clone(self) -> "GameState":
        g = GameState(
            num_players=self.num_players,
            bank=dict(self.bank),
            decks={t: list(d) for t, d in self.decks.items()},
            board={t: list(s) for t, s in self.board.items()},
            nobles=list(self.nobles),
            players=[p.clone() for p in self.players],
            to_move=self.to_move,
            ply=self.ply,
            winner=self.winner,
        )
        return g

    # ---- queries -------------------------------------------------------
    @property
    def current(self) -> PlayerState:
        return self.players[self.to_move]

    def is_over(self) -> bool:
        return self.winner is not None

    def board_card_ids(self) -> list[str]:
        out = []
        for tier in (1, 2, 3):
            for cid in self.board[tier]:
                if cid is not None:
                    out.append(cid)
        return out

    # ---- presentation --------------------------------------------------
    def view_state(self, perspective: int | None = None) -> dict:
        """JSON-serializable view for the UI, masking hidden reserved cards."""
        def card_dict(cid: str) -> dict:
            c = CARDS[cid]
            return {"id": c.id, "tier": c.tier, "bonus": c.bonus,
                    "points": c.points, "cost": dict(c.cost)}

        board = {}
        for tier in (1, 2, 3):
            board[f"L{tier}"] = [
                {**card_dict(cid), "slot": i} if cid is not None else None
                for i, cid in enumerate(self.board[tier])
            ]

        players_view = []
        for idx, p in enumerate(self.players):
            reserved_view = []
            for cid, hidden in zip(p.reserved, p.reserved_hidden):
                visible = (not hidden) or (perspective is None) or (perspective == idx)
                if visible:
                    reserved_view.append(card_dict(cid))
                else:
                    reserved_view.append({"hidden": True, "tier": CARDS[cid].tier})
            players_view.append({
                "id": p.id, "name": p.name, "type": p.ptype,
                "points": p.points,
                "tokens": dict(p.tokens),
                "bonuses": dict(p.bonuses),
                "cards": list(p.cards),
                "reserved": reserved_view,
                "reservedCount": len(p.reserved),
                "nobles": list(p.nobles),
            })

        return {
            "numPlayers": self.num_players,
            "toMove": self.to_move,
            "ply": self.ply,
            "bank": dict(self.bank),
            "nobles": [
                {"id": nid, "points": NOBLES[nid].points,
                 "requirement": dict(NOBLES[nid].requirement)}
                for nid in self.nobles
            ],
            "board": board,
            "deckCounts": {f"L{t}": len(self.decks[t]) for t in (1, 2, 3)},
            "players": players_view,
            "winner": self.winner,
        }
