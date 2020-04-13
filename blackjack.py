import numpy as np
import pandas as pd

from itertools import product
from functools import reduce


# adapted from Laurens Janssen
# https://gist.github.com/iiLaurens/ba9c479e71ee4ceef816ad50b87d9ebd

class BlackJack():

    def __init__(self, card_list=[2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11],
                 dealer_skip=17):
        self.ACTIONLIST = {
            0: 'skip',
            1: 'draw'
        }

        self.CARDS = np.array(card_list)
        self.BLACKJACK = 21
        self.DEALER_SKIP = dealer_skip
        self.STARTING_CARDS_PLAYER = 2
        self.STARTING_CARDS_DEALER = 1
        self.current_state = 0

        STATELIST = {0: (0, 0, 0)}  # Game start state
        self.STATELIST = {**STATELIST, **{nr + 1: state for nr, state in enumerate(
            product(range(2), range(self.CARDS.min() * self.STARTING_CARDS_PLAYER, self.BLACKJACK + 2),
                    range(self.CARDS.min() * self.STARTING_CARDS_DEALER, self.BLACKJACK + 2)))}}

        T = np.zeros((len(self.ACTIONLIST), len(self.STATELIST), len(self.STATELIST)))
        for a, i, j in product(self.ACTIONLIST.keys(), self.STATELIST.keys(), self.STATELIST.keys()):
            T[a, i, j] = self.blackjack_probability(a, i, j)

        # Define reward matrix
        R = np.zeros(len(self.STATELIST))
        for s in self.STATELIST.keys():
            R[s] = self.blackjack_rewards(s)
        # for a, s in product(self.ACTIONLIST.keys(), self.STATELIST.keys()):
        #    R[s, a] = self.blackjack_rewards(a, s)

        # Check that we have a valid transition matrix with transition probabilities summing to 1
        assert (T.sum(axis=2).round(10) == 1).all()
        self.T = T
        self.R = R

        # deal first card
        self.step(1)

    def cartesian(self, x, y):
        return np.dstack(np.meshgrid(x, y)).reshape(-1, 2).sum(axis=1)

    def deal_card_probability(self, count_now, count_next, take=1):
        if take > 1:
            cards = reduce(self.cartesian, [self.CARDS] * take)
        else:
            cards = self.CARDS

        return (np.minimum(count_now + cards, self.BLACKJACK + 1) == count_next).sum() / len(cards)

    def is_gameover(self, skipped, player, dealer):
        return any([
            dealer >= self.DEALER_SKIP and skipped == 1,
            dealer > self.BLACKJACK and skipped == 1,
            player > self.BLACKJACK
        ])

    def blackjack_probability(self, action, stateid_now, stateid_next):
        skipped_now, player_now, dealer_now = self.STATELIST[stateid_now]
        skipped_next, player_next, dealer_next = self.STATELIST[stateid_next]

        if stateid_now == stateid_next:
            # Game cannot stay in current state
            return 0.0

        if stateid_now == 0:
            if skipped_next == 1:
                # After start of the game the game cannot be in a skipped state
                return 0
            else:
                # State lower or equal than 1 is a start of a new game
                dealer_prob = self.deal_card_probability(0, dealer_next, take=self.STARTING_CARDS_DEALER)
                player_prob = self.deal_card_probability(0, player_next, take=self.STARTING_CARDS_PLAYER)

                return dealer_prob * player_prob

        if self.is_gameover(skipped_now, player_now, dealer_now):
            # We arrived at end state, now reset game
            return 1.0 if stateid_next == 0 else 0.0

        if skipped_now == 1:
            if skipped_next == 0 or player_next != player_now:
                # Once you skip you keep on skipping in blackjack
                # Also player cards cannot increase once in a skipped state
                return 0.0

        if self.ACTIONLIST[action] == 'skip' or skipped_now == 1:
            # If willingly skipped or in forced skip (attempted draw in already skipped game):
            if skipped_next != 1 or player_now != player_next:
                # Next state must be a skipped state with same card count for player
                return 0.0

        if self.ACTIONLIST[action] == 'draw' and skipped_now == 0 and skipped_next != 0:
            # Next state must be a drawable state
            return 0.0

        if dealer_now != dealer_next and player_now != player_next:
            # Only the player or the dealer can draw a card. Not both simultaneously!
            return 0.0

        # Now either the dealer or the player draws a card
        if self.ACTIONLIST[action] == 'draw' and skipped_now == 0:
            # Player draws a card
            prob = self.deal_card_probability(player_now, player_next, take=1)
        else:
            # Dealer draws a card
            if dealer_now >= self.DEALER_SKIP:
                if dealer_now != dealer_next:
                    # Dealer always stands once it has a card count higher than set amount
                    return 0.0
                else:
                    # Dealer stands
                    return 1.0

            prob = self.deal_card_probability(dealer_now, dealer_next, take=1)

        return prob

    # def blackjack_rewards(self, action, stateid):
    def blackjack_rewards(self, stateid):
        skipped, player, dealer = self.STATELIST[stateid]

        if not self.is_gameover(skipped, player, dealer):
            return 0
        elif player > self.BLACKJACK or (player <= dealer and dealer <= self.BLACKJACK):
            return -1
        elif player == self.BLACKJACK and dealer < self.BLACKJACK:
            return 1.5
        elif player > dealer or dealer > self.BLACKJACK:
            return 1
        else:
            raise Exception(f'Undefined reward: {skipped}, {player}, {dealer}')

    def get_matrices(self):
        # Define transition matrix

        return self.T, self.R

    def reset(self):
        # self = self.__init__(self.CARDS, self.DEALER_SKIP)
        self.current_state = 0
        self.step(1)
        return self.current_state

    def step(self, action):
        """take an action
        randomly draw next state
        return state, reward, and done
        if action == 0, then just keep looping through
        dealer actions until game is over
        """
        turn_continue = True
        while turn_continue:
            prob = np.random.random()
            probs = self.T[action, self.current_state, :]
            # print(prob)
            # print(probs)
            # print(probs.sum())
            # print('where')
            # print(np.where(probs > prob))
            probs = probs.cumsum()
            new_state = np.where(probs >= prob)[0][0]
            done = self.is_gameover(*self.STATELIST[new_state])
            self.current_state = new_state
            # print('new state', self.STATELIST[self.current_state])
            turn_continue = (action == 0) and (not done)
        reward = self.R[new_state]
        return new_state, reward, done

    def print_blackjack_policy(self, policy):
        idx = pd.MultiIndex.from_tuples(list(self.STATELIST.values()), names=['Skipped', 'Player', 'Dealer'])
        S = pd.Series(['x' if i == 1 else '.' for i in policy], index=idx)
        S = S.loc[S.index.get_level_values('Skipped') == 0].reset_index('Skipped', drop=True)
        S = S.loc[S.index.get_level_values('Player') > 0]
        S = S.loc[S.index.get_level_values('Dealer') > 0]
        return S.unstack(-1)

    def print_blackjack_rewards(self):
        idx = pd.MultiIndex.from_tuples(list(self.STATELIST.values()), names=['Skipped', 'Player', 'Dealer'])
        S = pd.Series(self.R[:, 0], index=idx)
        S = S.loc[S.index.get_level_values('Skipped') == 1].reset_index('Skipped', drop=True)
        S = S.loc[S.index.get_level_values('Player') > 0]
        S = S.loc[S.index.get_level_values('Dealer') > 0]
        return S.unstack(-1)