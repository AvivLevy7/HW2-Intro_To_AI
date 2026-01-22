from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time


# TODO: section a : 3

def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    other_robot = env.get_robot((robot_id + 1) % 2)
   
    def getdistopac(cur_robot):
        if cur_robot.package is not None:
            return (manhattan_distance(cur_robot.position,cur_robot.package.destination),0)
        my_cost = float("inf")
        for package in env.packages[0:2]:
            cost = manhattan_distance(cur_robot.position,package.position) + manhattan_distance(package.position,package.destination)
            if my_cost > cost:
                my_cost = cost
                dst = package.position
        return (my_cost,dst)
    def getdistocharge(cur_robot):
        my_cost = float("inf")
        for charge in env.charge_stations:
            cost = manhattan_distance(cur_robot.position,charge.position)
            my_cost = my_cost if my_cost < cost else cost
        return (-1)*my_cost
    credit_diff = robot.credit - other_robot.credit
    dst_my_nx_pac, dstrob = getdistopac(robot)
    dst_other_nx_pac,dstother = getdistopac(other_robot)
    dst_my_charge = getdistocharge(robot) if robot.battery < 5 and robot.credit > 0 else 0
    battery_diff = robot.battery - other_robot.battery
    dst_diff = dst_other_nx_pac - dst_my_nx_pac
    adv = -dst_my_nx_pac if robot.package is None else 0
    return 10*credit_diff + dst_diff + dst_my_charge + 10*battery_diff + 0.5*adv



class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        deadline = start_time + 0.99 * time_limit

        def RB_minimax(state: WarehouseEnv, current_id: int, depth: int) -> float:
            if deadline <= time.time() or depth <= 0 or state.done():
                return smart_heuristic(state, agent_id)

            operators, children = self.successors(state, current_id)
            if not operators:
                return smart_heuristic(state, agent_id)

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
                return smart_heuristic(state, agent_id)

            operators, children = self.successors(state, current_id)
            if not operators:
                return smart_heuristic(state, agent_id)

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
        depth = 1
        while deadline > time.time():
            operators, children = self.successors(env, agent_id)
            if not operators:
                break

            local_best_op = operators[0]
            local_best_val = float("-inf")
            for op, child in zip(operators, children):
                val = RB_alphabeta(child, (agent_id + 1) % 2, depth - 1, float("-inf"), float("inf"))
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
        start_time = time.time()
        deadline = start_time + 0.99 * time_limit

        def RB_expectimax(state: WarehouseEnv, current_id: int, depth: int) -> float:
            if deadline <= time.time() or depth <= 0 or state.done():
                return smart_heuristic(state, agent_id)

            operators, children = self.successors(state, current_id)
            if not operators:
                return smart_heuristic(state, agent_id)
            def operator_val(operat):
                return 3 if (operat == "move west" or operat == "pick up") else 1

            other_id = (current_id + 1) % 2

            if current_id == agent_id:
                cur_max = float("-inf")
                for child in children:
                    cur_max = max(cur_max, RB_expectimax(child, other_id, depth - 1))
                    if deadline <= time.time():
                        break
                return cur_max
            else:
                values = [operator_val(operat) for operat in operators]
                sum_of_vals = sum(values)
                if sum_of_vals == 0:
                    return smart_heuristic(state,agent_id)
                exp_max = 0
                for vals, child in zip(values, children):
                    exp_max += (vals / sum_of_vals)*RB_expectimax(child,other_id,depth-1)
                    if deadline <= time.time():
                        break
                return exp_max

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
                val = RB_expectimax(child, (agent_id + 1) % 2, depth)
                if val > local_best_val:
                    local_best_val = val
                    local_best_op = op
                if deadline <= time.time():
                    break

            best_op = local_best_op
            depth += 1

        return best_op


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
