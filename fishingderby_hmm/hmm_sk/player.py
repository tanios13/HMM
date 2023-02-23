from player_controller_hmm import PlayerControllerHMMAbstract
from Baum_Welch import *
from constants import *
import time

class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """
        N_STATES = 4

        self.A_matrixes = [[[0 for _ in range(N_STATES)] for _ in range(N_STATES)] for _ in range(N_SPECIES)]
        self.B_matrixes = [[[0 for _ in range(N_EMISSIONS)] for _ in range(N_STATES)] for _ in range(N_SPECIES)]
        self.pi_matrixes = [[[0 for _ in range(N_STATES)]] for _ in range(N_SPECIES)]
        self.observations = [[0 for _ in range(N_STEPS)] for _ in range(N_FISH)]
        self.fish_index = 0
        self.step = 0

        self.fish_species = [-1] * N_FISH

        prob_Pi = 1 / N_STATES
        for f in range(N_SPECIES):
            for i in range(N_STATES):
                if i % 2 == 0:
                    self.pi_matrixes[f][0][i] = prob_Pi - N_STATES / 100
                else:
                    self.pi_matrixes[f][0][i] = prob_Pi + N_STATES / 100

        prob_A = 1 / N_STATES
        for f in range(N_SPECIES):
            for i in range(N_STATES):
                for j in range(N_STATES):
                    if j % 2 == 0:
                        self.A_matrixes[f][i][j] = prob_A - N_STATES / 100
                    else:
                        self.A_matrixes[f][i][j] = prob_A + N_STATES / 100

        prob_B = 1 / N_EMISSIONS
        for f in range(N_SPECIES):
            for i in range(N_STATES):
                for j in range(N_EMISSIONS):
                    if j % 2 == 0:
                        self.B_matrixes[f][i][j] = prob_B - N_EMISSIONS / 100
                    else:
                        self.B_matrixes[f][i][j] = prob_B + N_EMISSIONS / 100



    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        self.step = step

        # add the new observations to the model
        for f in range(N_FISH):
            self.observations[f][step] = observations[f]

        if step >= 90 and self.fish_index < N_FISH:
            fish_index = self.fish_index
            # compute the probability of the observation of the fish with the model of each species
            prob = 0
            proba_of_obs_seq = []
            species = 0
            for i in range(N_SPECIES):
                proba_of_obs_seq.append(compute_prob_obs(self.A_matrixes[i], self.B_matrixes[i], self.pi_matrixes[i], self.observations[fish_index], step))

            self.fish_index += 1
            return fish_index, proba_of_obs_seq.index(max(proba_of_obs_seq))
        else:
            # just wait to have more observations for better estimation of the model
            return None


    def reveal(self, correct, fish_id, true_type):
            """
            This methods gets called whenever a guess was made.
            It informs the player about the guess result
            and reveals the correct type of that fish.
            :param correct: tells if the guess was correct
            :param fish_id: fish's index
            :param true_type: the correct type of the fish
            :return:
            """
            N_STATES = 4

            self.fish_species[fish_id] = true_type
            if correct is True:
                return
            else:
                try:
                    self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type] = \
                        Baum_Welch_algorithm(self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type],
                                             self.observations[fish_id], self.step, 0, time.time())
                except ZeroDivisionError:

                    prob_Pi = 1 / N_STATES
                    for i in range(N_STATES):
                        if i % 2 == 0:
                            self.pi_matrixes[true_type][0][i] = prob_Pi - N_STATES / 100
                        else:
                            self.pi_matrixes[true_type][0][i] = prob_Pi + N_STATES / 100

                    prob_A = 1 / N_STATES
                    for i in range(N_STATES):
                        for j in range(N_STATES):
                            if j % 2 == 0:
                                self.A_matrixes[true_type][i][j] = prob_A - N_STATES / 100
                            else:
                                self.A_matrixes[true_type][i][j] = prob_A + N_STATES / 100

                    prob_B = 1 / N_EMISSIONS
                    for i in range(N_STATES):
                        for j in range(N_EMISSIONS):
                            if j % 2 == 0:
                                self.B_matrixes[true_type][i][j] = prob_B - N_EMISSIONS / 100
                            else:
                                self.B_matrixes[true_type][i][j] = prob_B + N_EMISSIONS / 100

                    self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type] = \
                        Baum_Welch_algorithm(self.A_matrixes[true_type], self.B_matrixes[true_type], self.pi_matrixes[true_type],
                                             self.observations[fish_id], self.step, 0, time.time())

