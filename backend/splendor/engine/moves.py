"""Move representation, legal-move generation, and state transitions.

A ``Move`` describes a whole turn: a primary action (take3 / take2 / reserve /
purchase) plus any forced follow-ups that the turn resolves (discarding down to
the 10-token limit, and visiting a noble). Keeping a turn = one move keeps game
records at one ply per turn, which makes replay and analysis clean.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from splendor.engine.cards import CARDS, NOBLES, COLORS, GOLD, ALL_TOKENS
from splendor.engine import rules
from splendor.engine.state import GameState, PlayerState


@dataclass
class Move:
    type: str                                   # take3 | take2 | reserve | purchase
    gems: tuple = ()                            # take3: distinct colors
    color: str = ""                            # take2: the doubled color
    card_id: str = ""                          # reserve/purchase target identity
    source: str = ""                           # board | deck | reserve
    tier: int = 0                               # board/deck location
    slot: int = -1                              # board slot
    reserve_index: int = -1                     # purchase-from-reserve index
    discard: dict = field(default_factory=dict)  # color -> count returned to bank
    noble: str | None = None                    # noble visited this turn

    def key(self) -> tuple:
        """Hashable identity for tree-search dedup / lookup.

        A deck-top reserve is keyed by the *decision* (tier) only: the drawn
        card is hidden chance, so it must aggregate across determinizations and
        match the true move during replay/selection.
        """
        card_id = "" if (self.type == "reserve" and self.source == "deck") else self.card_id
        return (self.type, self.gems, self.color, card_id, self.source,
                self.tier, self.slot, self.reserve_index,
                tuple(sorted(self.discard.items())), self.noble)


# ---------------------------------------------------------------------------
# Payment
# ---------------------------------------------------------------------------
def payment_for(player: PlayerState, card_id: str):
    """Minimal token payment (gold last) for ``player`` to buy ``card_id``.

    Returns a color->count dict (may include GOLD) or ``None`` if unaffordable.
    """
    card = CARDS[card_id]
    pay = {}
    gold_needed = 0
    for color, amount in card.cost.items():
        effective = amount - player.bonuses[color]
        if effective <= 0:
            continue
        use = min(effective, player.tokens[color])
        if use:
            pay[color] = use
        gold_needed += effective - use
    if gold_needed > player.tokens[GOLD]:
        return None
    if gold_needed:
        pay[GOLD] = gold_needed
    return pay


def can_afford(player: PlayerState, card_id: str) -> bool:
    return payment_for(player, card_id) is not None


# ---------------------------------------------------------------------------
# Noble qualification
# ---------------------------------------------------------------------------
def qualifying_nobles(state: GameState, player: PlayerState) -> list[str]:
    out = []
    for nid in state.nobles:
        req = NOBLES[nid].requirement
        if all(player.bonuses[c] >= n for c, n in req.items()):
            out.append(nid)
    return out


# ---------------------------------------------------------------------------
# Discard enumeration (only when a take/reserve pushes above the cap)
# ---------------------------------------------------------------------------
def _discard_options(tokens_after: dict, excess: int) -> list[dict]:
    colors = [c for c in ALL_TOKENS if tokens_after[c] > 0]
    results: list[dict] = []

    def rec(i: int, remaining: int, acc: dict):
        if remaining == 0:
            results.append(dict(acc))
            return
        if i >= len(colors):
            return
        c = colors[i]
        hi = min(tokens_after[c], remaining)
        for take in range(hi, -1, -1):
            if take:
                acc[c] = take
            rec(i + 1, remaining - take, acc)
            if take:
                del acc[c]

    rec(0, excess, {})
    return results


# ---------------------------------------------------------------------------
# Legal move generation
# ---------------------------------------------------------------------------
def legal_moves(state: GameState) -> list[Move]:
    if state.is_over():
        return []
    p = state.current
    bank = state.bank
    moves: list[Move] = []

    # --- primary actions that gain tokens (need discard/noble expansion) ---
    primaries: list[tuple[Move, dict]] = []  # (move, tokens_after_dict)

    # take 3 different colors (or fewer only if fewer are available)
    available = [c for c in COLORS if bank[c] > 0]
    if len(available) >= 3:
        combos = itertools.combinations(available, 3)
    elif available:
        combos = [tuple(available)]
    else:
        combos = []
    for combo in combos:
        ta = dict(p.tokens)
        for c in combo:
            ta[c] += 1
        primaries.append((Move(type="take3", gems=tuple(combo)), ta))

    # take 2 of one color (pile must be >= 4)
    for c in COLORS:
        if bank[c] >= rules.TAKE_SAME_MIN_PILE:
            ta = dict(p.tokens)
            ta[c] += 2
            primaries.append((Move(type="take2", color=c), ta))

    # reserve (board or deck top); gains a gold if available
    if len(p.reserved) < rules.MAX_RESERVED:
        gold_gain = 1 if bank[GOLD] > 0 else 0
        for tier in (1, 2, 3):
            for slot, cid in enumerate(state.board[tier]):
                if cid is None:
                    continue
                ta = dict(p.tokens)
                ta[GOLD] += gold_gain
                primaries.append(
                    (Move(type="reserve", card_id=cid, source="board",
                          tier=tier, slot=slot), ta))
            if state.decks[tier]:
                top = state.decks[tier][-1]
                ta = dict(p.tokens)
                ta[GOLD] += gold_gain
                primaries.append(
                    (Move(type="reserve", card_id=top, source="deck", tier=tier), ta))

    for base, tokens_after in primaries:
        total = sum(tokens_after.values())
        excess = total - rules.MAX_TOKENS_IN_HAND
        discards = _discard_options(tokens_after, excess) if excess > 0 else [{}]
        for disc in discards:
            m = Move(type=base.type, gems=base.gems, color=base.color,
                     card_id=base.card_id, source=base.source, tier=base.tier,
                     slot=base.slot, discard=disc)
            moves.append(m)

    # --- purchases (reduce tokens; never need discard) ---
    purchase_targets = []  # (card_id, source, tier, slot, reserve_index)
    for tier in (1, 2, 3):
        for slot, cid in enumerate(state.board[tier]):
            if cid is not None:
                purchase_targets.append((cid, "board", tier, slot, -1))
    for i, cid in enumerate(p.reserved):
        purchase_targets.append((cid, "reserve", 0, -1, i))

    for cid, source, tier, slot, ridx in purchase_targets:
        if not can_afford(p, cid):
            continue
        base = Move(type="purchase", card_id=cid, source=source,
                    tier=tier, slot=slot, reserve_index=ridx)
        # bonuses gained -> possibly a new noble; enumerate choices
        bonus = CARDS[cid].bonus
        saved = p.bonuses[bonus]
        p.bonuses[bonus] += 1
        nobles = qualifying_nobles(state, p)
        p.bonuses[bonus] = saved
        if len(nobles) <= 1:
            base.noble = nobles[0] if nobles else None
            moves.append(base)
        else:
            for nid in nobles:
                m = Move(type="purchase", card_id=cid, source=source, tier=tier,
                         slot=slot, reserve_index=ridx, noble=nid)
                moves.append(m)

    return moves


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def _take_board_card(state: GameState, tier: int, slot: int) -> str:
    cid = state.board[tier][slot]
    deck = state.decks[tier]
    state.board[tier][slot] = deck.pop() if deck else None
    return cid


def apply_move(state: GameState, move: Move) -> None:
    """Advance ``state`` in place by one full turn."""
    p = state.current
    bank = state.bank

    if move.type == "take3":
        for c in move.gems:
            bank[c] -= 1
            p.tokens[c] += 1
    elif move.type == "take2":
        bank[move.color] -= 2
        p.tokens[move.color] += 2
    elif move.type == "reserve":
        if move.source == "board":
            cid = _take_board_card(state, move.tier, move.slot)
            hidden = False
        else:  # deck
            cid = state.decks[move.tier].pop()
            hidden = True
        p.reserved.append(cid)
        p.reserved_hidden.append(hidden)
        if bank[GOLD] > 0:
            bank[GOLD] -= 1
            p.tokens[GOLD] += 1
    elif move.type == "purchase":
        pay = payment_for(p, move.card_id)
        if pay is None:
            raise ValueError(f"illegal purchase: {move.card_id} unaffordable")
        for color, amt in pay.items():
            p.tokens[color] -= amt
            bank[color] += amt
        if move.source == "board":
            _take_board_card(state, move.tier, move.slot)
        else:  # reserve
            idx = move.reserve_index
            del p.reserved[idx]
            del p.reserved_hidden[idx]
        card = CARDS[move.card_id]
        p.cards.append(card.id)
        p.bonuses[card.bonus] += 1
        p.points += card.points
    else:
        raise ValueError(f"unknown move type: {move.type}")

    # discard excess tokens
    for color, cnt in move.discard.items():
        p.tokens[color] -= cnt
        bank[color] += cnt

    # noble visit
    nid = move.noble
    if nid is None and move.type == "purchase":
        q = qualifying_nobles(state, p)
        nid = q[0] if len(q) == 1 else None
    if nid is not None and nid in state.nobles:
        state.nobles.remove(nid)
        p.nobles.append(nid)
        p.points += NOBLES[nid].points

    # advance turn & check for game end at the close of a full round
    state.ply += 1
    state.to_move = (state.to_move + 1) % state.num_players
    if state.to_move == 0:
        leader = _round_winner(state)
        if leader is not None:
            state.winner = leader


def _round_winner(state: GameState) -> int | None:
    """If any player has reached the points goal, return the winner index."""
    if not any(pl.points >= rules.WINNING_POINTS for pl in state.players):
        return None
    best = None
    for idx, pl in enumerate(state.players):
        # higher points, then fewer development cards
        key = (pl.points, -len(pl.cards))
        if best is None or key > best[1]:
            best = (idx, key)
    return best[0]


def find_legal(state: GameState, move: Move) -> Move | None:
    """Return the canonical legal Move matching ``move`` (by key), or None."""
    target = move.key()
    for m in legal_moves(state):
        if m.key() == target:
            return m
    return None
