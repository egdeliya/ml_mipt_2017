import numpy as np
import sys
from six import StringIO

from gym import spaces, utils
from gym.envs.toy_text import discrete

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

class TaxiEnvUpgrV2(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich

    rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP,dtype='c')

        self.locs = locs = [(0,0), (0,4), (4,0), (4,3)]

        nS = 500
        nR = 5
        nC = 5
        maxR = nR-1
        maxC = nC-1
        isd = np.zeros(nS)
        nA = 6
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for row in range(5):
            for col in range(5):
                for passidx in range(5):
                    for destidx in range(4):
                        state = self.encode(row, col, passidx, destidx)
                        if passidx < 4 and passidx != destidx:
                            isd[state] += 1
                        for a in range(nA):
                            # defaults
                            newrow, newcol, newpassidx = row, col, passidx
                            reward = -1
                            done = False
                            taxiloc = (row, col)

                            if a==0: # South
                                newrow = min(row+1, maxR)
                                # добавим награду за уменьшение расстояния
                            elif a==1: # North
                                newrow = max(row-1, 0)
                            if a==2 and self.desc[1+row,2*col+2]==b":": # West
                                newcol = min(col+1, maxC)
                            elif a==3 and self.desc[1+row,2*col]==b":": # East
                                newcol = max(col-1, 0)
                            elif a==4: # pickup
                                if (passidx < 4 and taxiloc == locs[passidx]):
                                    newpassidx = 4
                                    reward = 10
                                else:
                                    reward -= 10
                            elif a==5: # dropoff
                                if (taxiloc == locs[destidx]) and passidx==4:
                                    done = True
                                    reward = 10
                                elif (taxiloc in locs) and passidx==4:
                                    newpassidx = locs.index(taxiloc)
                                else:
                                    reward = -10
                            reward += self.evaluate_reward(a, passidx, destidx, newrow, newcol)
                            newstate = self.encode(newrow, newcol, newpassidx, destidx)
                            P[state][a].append((1.0, newstate, reward, done))
        isd /= isd.sum()
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
        
    def evaluate_reward(self, action, passidx, destidx, row, col):
        
        reward = 0
        if (passidx != 4):
            target_row, target_col = self.locs[passidx]
        else:
            target_row, target_col = self.locs[destidx]
            
        if (1 <= row <= 2):
            reward -= (abs(target_row - row) + abs(target_col - col))
        elif (row == 0):
            if (target_row == 0):
                if (col == target_col):
                    pass
                else:
                    distance = target_col - col
                    if (distance < 0):
                        if (col >= 2):
                            reward -= (2 + abs(distance))
                        else:
                            reward -= abs(distance)
                    else:
                        if (col < 2):
                            reward -= (2 + abs(distance))
                        else:
                            reward -= abs(distance)
            elif (target_row == 4):
                reward -= (abs(target_row - row) + abs(target_col - col))
        elif (3 <= row <= 4):
            if (target_row == 4):
                if (col == target_col):
                    pass
                else:
                    hor_distance = target_col - col
                    vert_distance = target_row - row
                    if (hor_distance < 0):
                        if (col == 4):
                            reward -= (abs(vert_distance) + abs(hor_distance))
                        else:
                            reward -= ((4-abs(vert_distance)) + abs(hor_distance))
                    elif (hor_distance == 0):
                        reward -= abs(vert_distance)
                    elif (hor_distance > 0):
                        reward -= ((4-abs(vert_distance)) + abs(hor_distance))
                            
            elif (target_row == 0):
                reward -= (abs(target_row - row) + abs(target_col - col))
                    
        return reward

    def encode(self, taxirow, taxicol, passloc, destidx):
        # (5) 5, 5, 4
        i = taxirow
        i *= 5
        i += taxicol
        i *= 5
        i += passloc
        i *= 4
        i += destidx
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxirow, taxicol, passidx, destidx = self.decode(self.s)
        def ul(x): return "_" if x == " " else x
        if passidx < 4:
            out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
            pi, pj = self.locs[passidx]
            out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
        else: # passenger in taxi
            out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

        di, dj = self.locs[destidx]
        out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out])+"\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            return outfile
