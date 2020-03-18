import numpy as np


class BirthDeathMH(object):
    """
    This cass implements Aglorithm 7.5
    """
    def __init__(
            self, papangelou, birth_proposal_rng, birth_proposal_dens,
            update_proposal_rng, update_proposal_dens):
        self.papangelou = papangelou
        self.birth_proposal_rng = birth_proposal_rng
        self.birth_proposal_dens = birth_proposal_dens
        self.update_proposal_rng = update_proposal_rng
        self.update_proposal_dens = update_proposal_dens
        self.pbirth = 0.9
        self.state = np.zeros((10, 2))

    def birth_move(self):
        csi = self.birth_proposal_rng(self.state)
        prop = np.vstack([self.state, [csi]])
        arate = self.papangelou(csi, self.state) + (1 - self.pbirth) - \
            np.log(prop.shape[0]) - \
            self.pbirth - self.birth_proposal_dens(csi, self.state)

        if np.log(np.random.uniform()) < arate:
            self.state = prop

    def death_move(self):
        npoints = self.state.shape[0]
        
        ind = np.random.choice(npoints)
        prop = np.delete(self.state.copy(), ind, axis=0)

        arate = self.pbirth + \
            self.birth_proposal_dens(self.state[ind, :], prop) - \
            self.papangelou(self.state[ind, :], self.state) - \
            (1 - self.pbirth) + np.log(npoints)

        if np.log(np.random.uniform()) < arate:
            self.state = prop.copy()

    def update_move(self):
        npoints = self.state.shape[0]
        ind = np.random.choice(npoints)
        csi = self.update_proposal_rng(self.state, ind)

        # compute acceptance ratio
        prop = self.state.copy()
        aux = np.delete(self.state, ind, axis=0)
        prop[ind] = csi
        arate = self.papangelou(csi, aux, log=True) + \
            self.update_proposal_dens(prop, self.state, ind, log=True) - \
            self.papangelou(self.state[ind, :], aux, log=True) - \
            self.update_proposal_dens(self.state, self.state, ind, log=True)

        if np.log(np.random.uniform()) < arate:
            self.state = prop

    def run_one(self, q):
        if np.random.uniform() < q and self.state.shape[0] > 0:
            self.update_move()
        else:
            if np.random.uniform () < self.pbirth:
                self.birth_move()
            elif self.state.shape[0] > 0:
                self.death_move()

    def run(self,  nburn, nsamples, init_state, q=0.5):
        out = [None] * nsamples
        self.state = init_state

        for i in range(nburn):
            self.run_one(q)

        for i in range(nsamples):
            self.run_one(q)
            out[i] = self.state

        return out
