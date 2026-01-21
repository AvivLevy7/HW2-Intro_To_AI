from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    pass

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        deadline = start_time + 0.99 * time_limit

        def RB_minimax(state: WarehouseEnv, current_id: int, depth: int) -> float:
            if deadline <= time.time() or depth <= 0 or state.done():
                return self.heuristic(state, agent_id)

            operators, children = self.successors(state, current_id)
            if not operators:
                return self.heuristic(state, agent_id)

            other_id = (current_id + 1) % 2

            if current_id == agent_id:
                cur_max = float("-inf")
                for child in children:
                    cur_max = max(cur_max, RB_minimax(child, other_id, depth - 1))
                    if deadline <= time.time():
                        break
                return cur_max
            else:
                cur_min = float("inf")
                for child in children:
                    cur_min = min(cur_min, RB_minimax(child, other_id, depth - 1))
                    if deadline <= time.time():
                        break
                return cur_min

        legal = env.get_legal_operators(agent_id)
        if not legal:
            return "park"

        best_op = legal[0]
        depth = 0
        while deadline > time.time():
            operators, children = self.successors(env, agent_id)
            if not operators:
                break

            local_best_op = operators[0]
            local_best_val = float("-inf")
            for op, child in zip(operators, children):
                val = RB_minimax(child, (agent_id + 1) % 2, depth)
                if val > local_best_val:
                    local_best_val = val
                    local_best_op = op
                if deadline <= time.time():
                    break

            best_op = local_best_op
            depth += 1

        return best_op


class AgentAlphaBeta(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        deadline = start_time + 0.99 * time_limit

        def RB_alphabeta(state: WarehouseEnv, current_id: int, depth: int, alpha: float, beta: float) -> float:
            if deadline <= time.time() or depth <= 0 or state.done():
                return self.heuristic(state, agent_id)

            operators, children = self.successors(state, current_id)
            if not operators:
                return self.heuristic(state, agent_id)

            other_id = (current_id + 1) % 2

            if current_id == agent_id:
                cur_max = float("-inf")
                for child in children:
                    cur_max = max(cur_max, RB_alphabeta(child, other_id, depth - 1, alpha, beta))
                    alpha = max(alpha, cur_max)
                    if deadline <= time.time():
                        break
                    if cur_max >= beta:
                        return float("inf")
                return cur_max
            else:
                cur_min = float("inf")
                for child in children:
                    cur_min = min(cur_min, RB_alphabeta(child, other_id, depth - 1, alpha, beta))
                    beta = min(beta, cur_min)
                    if deadline <= time.time():
                        break
                    if cur_min <= alpha:
                        return float("-inf")
                return cur_min

        legal = env.get_legal_operators(agent_id)
        if not legal:
            return "park"

        best_op = legal[0]
        depth = 0
        while deadline > time.time():
            operators, children = self.successors(env, agent_id)
            if not operators:
                break

            local_best_op = operators[0]
            local_best_val = float("-inf")
            for op, child in zip(operators, children):
                val = RB_alphabeta(child, (agent_id + 1) % 2, depth, float("-inf"), float("inf"))
                if val > local_best_val:
                    local_best_val = val
                    local_best_op = op
                if deadline <= time.time():
                    break

            best_op = local_best_op
            depth += 1

        return best_op


class AgentExpectimax(Agent):
    # TODO: section d : 3
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)