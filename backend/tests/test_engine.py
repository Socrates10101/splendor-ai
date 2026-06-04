import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splendor.engine.cards import CARDS, NOBLES, CARDS_BY_TIER, COLORS, GOLD
from splendor.engine.state import GameState
from splendor.engine.moves import legal_moves, apply_move, payment_for, qualifying_nobles
from splendor.engine import rules

SPECS = [{"id": "p0", "name": "A"}, {"id": "p1", "name": "B"}]


def test_card_and_noble_counts():
    assert len(CARDS) == 90
    assert len(CARDS_BY_TIER[1]) == 40
    assert len(CARDS_BY_TIER[2]) == 30
    assert len(CARDS_BY_TIER[3]) == 20
    assert len(NOBLES) == 10
    # every card cost references only real colors
    for c in CARDS.values():
        assert c.bonus in COLORS
        assert all(k in COLORS for k in c.cost)


def test_initial_setup_two_players():
    g = GameState.new(SPECS, seed=42)
    for c in COLORS:
        assert g.bank[c] == 4
    assert g.bank[GOLD] == 5
    for tier in (1, 2, 3):
        assert sum(1 for s in g.board[tier] if s is not None) == 4
    assert len(g.nobles) == 3            # players + 1
    assert g.to_move == 0 and g.ply == 0
    # decks: tier sizes minus the 4 dealt
    assert len(g.decks[1]) == 36
    assert len(g.decks[2]) == 26
    assert len(g.decks[3]) == 16


def test_deterministic_setup():
    a = GameState.new(SPECS, seed=7).view_state()
    b = GameState.new(SPECS, seed=7).view_state()
    c = GameState.new(SPECS, seed=8).view_state()
    assert a == b
    assert a != c


def test_take3_transfers_tokens():
    g = GameState.new(SPECS, seed=1)
    moves = [m for m in legal_moves(g) if m.type == "take3" and len(m.gems) == 3]
    assert moves
    m = moves[0]
    before_bank = dict(g.bank)
    apply_move(g, m)
    p = g.players[0]
    for c in m.gems:
        assert p.tokens[c] == 1
        assert g.bank[c] == before_bank[c] - 1
    assert g.to_move == 1 and g.ply == 1


def test_take2_requires_pile_of_four():
    g = GameState.new(SPECS, seed=1)
    # 2-player banks start at 4, so take2 is legal for every color
    take2 = [m for m in legal_moves(g) if m.type == "take2"]
    assert len(take2) == len(COLORS)
    g.bank["r"] = 3
    take2 = [m for m in legal_moves(g) if m.type == "take2"]
    assert all(m.color != "r" for m in take2)


def test_reserve_grants_gold_and_hides_deck_card():
    g = GameState.new(SPECS, seed=3)
    deck_reserve = [m for m in legal_moves(g) if m.type == "reserve" and m.source == "deck"]
    assert deck_reserve
    apply_move(g, deck_reserve[0])
    p = g.players[0]
    assert p.tokens[GOLD] == 1
    assert len(p.reserved) == 1
    assert p.reserved_hidden[0] is True
    # opponent's view hides the deck-reserved card
    view = g.view_state(perspective=1)
    assert view["players"][0]["reserved"][0].get("hidden") is True


def test_payment_uses_bonuses_then_gold():
    g = GameState.new(SPECS, seed=1)
    p = g.players[0]
    # pick a tier-1 card and hand the player exactly its cost
    cid = g.board[1][0]
    card = CARDS[cid]
    for color, amt in card.cost.items():
        p.tokens[color] = amt
    pay = payment_for(p, cid)
    assert pay is not None
    assert all(pay.get(c, 0) == card.cost.get(c, 0) for c in card.cost)
    assert GOLD not in pay
    # now make one color short by one and give a gold
    short_color = next(iter(card.cost))
    p.tokens[short_color] -= 1
    p.tokens[GOLD] = 1
    pay = payment_for(p, cid)
    assert pay is not None and pay[GOLD] == 1


def test_bonus_discounts_cost():
    g = GameState.new(SPECS, seed=1)
    p = g.players[0]
    cid = g.board[1][0]
    card = CARDS[cid]
    color = next(iter(card.cost))
    p.bonuses[color] = card.cost[color]   # fully discount that color
    for c, amt in card.cost.items():
        if c != color:
            p.tokens[c] = amt
    pay = payment_for(p, cid)
    assert pay is not None
    assert pay.get(color, 0) == 0


def test_purchase_awards_points_and_bonus():
    g = GameState.new(SPECS, seed=1)
    p = g.players[0]
    cid = g.board[1][0]
    card = CARDS[cid]
    for color, amt in card.cost.items():
        p.tokens[color] = amt
    buy = next(m for m in legal_moves(g)
               if m.type == "purchase" and m.card_id == cid)
    apply_move(g, buy)
    assert cid in p.cards
    assert p.bonuses[card.bonus] == 1
    assert p.points == card.points


def test_token_cap_forces_discard():
    g = GameState.new(SPECS, seed=1)
    p = g.players[0]
    # give the player 9 tokens; a take3 -> 12 -> must discard 2
    p.tokens["w"] = 9
    g.bank["w"] = 4
    take3 = [m for m in legal_moves(g) if m.type == "take3" and len(m.gems) == 3]
    assert take3
    assert all(sum(m.discard.values()) == 2 for m in take3)
    apply_move(g, take3[0])
    assert p.total_tokens() == rules.MAX_TOKENS_IN_HAND


def test_noble_visit_on_qualifying_purchase():
    g = GameState.new(SPECS, seed=1)
    p = g.players[0]
    noble = NOBLES[g.nobles[0]]
    # satisfy all-but-one of the noble's requirement via bonuses
    color, need = next(iter(noble.requirement.items()))
    p.bonuses.update({c: n for c, n in noble.requirement.items()})
    p.bonuses[color] = need - 1
    # buy a board card of that color for free (zero its cost via bonuses)
    cid = next(c for c in g.board_card_ids() if CARDS[c].bonus == color)
    card = CARDS[cid]
    for c, amt in card.cost.items():
        p.tokens[c] = amt
    buy = next(m for m in legal_moves(g) if m.type == "purchase" and m.card_id == cid)
    assert buy.noble == noble.id
    apply_move(g, buy)
    assert noble.id in p.nobles
    assert p.points >= card.points + 3


def test_game_ends_after_round_when_target_reached():
    g = GameState.new(SPECS, seed=1)
    g.players[0].points = 16
    # p0 just hit target; game should not end until round returns to p0
    g.to_move = 0
    m = legal_moves(g)[0]
    apply_move(g, m)             # now p1 to move
    assert not g.is_over()
    m = legal_moves(g)[0]
    apply_move(g, m)             # round complete -> back to p0
    assert g.is_over()
    assert g.winner == 0
