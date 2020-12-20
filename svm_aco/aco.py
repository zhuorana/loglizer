import random
from loglizer.models import SVM
from loglizer import dataloader, preprocessing
import time


class ACO:
    def __init__(self):
        self.global_pheromone_matrix = []
        self.svm_training_result_matrix = []
        self.global_ant_position_matrix = []
        self.ants = []

    def init_matrix(self):
        # [[row[i] for row in matrix] for i in range(4)]
        self.global_pheromone_matrix = [[-1 for i in range(0, COLUMN + 2)] for j in range(0, ROW + 2)]
        self.svm_training_result_matrix = [[-1 for i in range(0, COLUMN + 2)] for j in range(0, ROW + 2)]
        self.global_ant_position_matrix = [[-1 for i in range(0, COLUMN + 2)] for j in range(0, ROW + 2)]

    def init_pheromone_and_training_matrix(self):
        for index_x in range(1, ROW + 1):
            for index_y in range(1, COLUMN + 1):
                self.global_pheromone_matrix[index_x][index_y] = 1
                self.svm_training_result_matrix[index_x][index_y] = -1
                self.global_ant_position_matrix[index_x][index_y] = 0
        print(self.global_pheromone_matrix)
        print(self.svm_training_result_matrix)
        print(self.global_ant_position_matrix)

    def init_ants(self):
        self.ants = [Ant() for i in range(0, NUM_ANTS)]
        for _ant in self.ants:
            _ant.allocate_ant(self.global_ant_position_matrix)
            x, y = _ant.get_position_x_y()
            # mark as occupied
            self.global_ant_position_matrix[x][y] = 1

    def pheromone_evaporate(self):
        for index_x in range(1, ROW + 1):
            for index_y in range(1, COLUMN + 1):
                self.global_pheromone_matrix[index_x][index_y] = \
                    self.global_pheromone_matrix[index_x][index_y] * EVAPORATE_FACTOR


class Ant:
    def __init__(self):
        self.visited = []
        self.current_position_x = 0
        self.current_position_y = 0
        # self.all_neighbours = []

    def allocate_ant(self, ant_position_matrix):
        temp_x = 0
        temp_y = 0
        while True:
            temp_x = random.randint(1, ROW)
            temp_y = random.randint(1, COLUMN)
            if ant_position_matrix[temp_x][temp_y] == -1:
                raise Exception("Sorry, you are at the ghost site")
            if ant_position_matrix[temp_x][temp_y] == 0:
                break

        self.current_position_x = temp_x
        self.current_position_y = temp_y
        # mark this position as occupied
        aco.global_ant_position_matrix[temp_x][temp_y] = 1
        self.visited.append([temp_x, temp_y])

    def move_to_next_position(self, _x, _y):
        if [_x, _y] not in self.visited:
            self.visited.append([_x, _y])
        aco.global_ant_position_matrix[self.current_position_x][self.current_position_y] = 0
        self.current_position_x = _x
        self.current_position_y = _y
        aco.global_ant_position_matrix[_x][_y] = 1

    def get_position_x_y(self):
        return self.current_position_x, self.current_position_y

    def get_all_possible_neighbour(self, ant_position_matrix):
        _x = self.current_position_x
        _y = self.current_position_y
        # start with top left corner and go clock-wise
        possible_neighbour_positions = []
        if ant_position_matrix[_x - 1][_y - 1] == 0:
            possible_neighbour_positions.append([_x - 1, _y - 1])
        # the position above it
        if ant_position_matrix[_x - 1][_y] == 0:
            possible_neighbour_positions.append([_x - 1, _y])
        # top right
        if ant_position_matrix[_x - 1][_y + 1] == 0:
            possible_neighbour_positions.append([_x - 1, _y + 1])
        # right
        if ant_position_matrix[_x][_y + 1] == 0:
            possible_neighbour_positions.append([_x, _y + 1])
        # bottom right
        if ant_position_matrix[_x + 1][_y + 1] == 0:
            possible_neighbour_positions.append([_x + 1, _y + 1])
        # bottom (below)
        if ant_position_matrix[_x + 1][_y] == 0:
            possible_neighbour_positions.append([_x + 1, _y])
        # bottom left
        if ant_position_matrix[_x + 1][_y - 1] == 0:
            possible_neighbour_positions.append([_x + 1, _y - 1])
        # left
        if ant_position_matrix[_x][_y - 1] == 0:
            possible_neighbour_positions.append([_x, _y - 1])
        result = []
        for neighbour in possible_neighbour_positions:
            if neighbour not in self.visited:
                result.append(neighbour)
        return result

    def get_next_move_by_pheromone(self, ant_position_matrix, pheromone_matrix):
        _x = self.current_position_x
        _y = self.current_position_y
        possible_neighbours = self.get_all_possible_neighbour(ant_position_matrix)
        if len(possible_neighbours) == 0:
            # todo: if length is 0, return current position
            print("no neighbours so don't move.")
            return self.current_position_x, self.current_position_y
        # calculate according to state transition rule
        sum_pheromone = 0
        for neighbour in possible_neighbours:
            sum_pheromone = sum_pheromone + pheromone_matrix[neighbour[0]][neighbour[1]]
        random_number = random.uniform(0, sum_pheromone)
        current_sum = 0
        for neighbour in possible_neighbours:
            current_sum = current_sum + pheromone_matrix[neighbour[0]][neighbour[1]]
            if random_number <= current_sum:
                return neighbour[0], neighbour[1]
        # todo: delete this shit
        raise Exception("shit")

    def __str__(self):
        return str(self.current_position_x) + " " + str(self.current_position_y) + " Visited: " + str(self.visited)

    def __eq__(self, other):
        if not isinstance(other, Ant):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return \
            self.current_position_x == other.current_position_x and self.current_position_y == other.current_position_y


ROW = 10
COLUMN = 10
NUM_ANTS = 7
C_VALUE_RANGE_STARTING_INDEX = -5
GAMMA_VALUE_RANGE_STARTING_INDEX = -5
MAX_ITERATION = 10
EVAPORATE_FACTOR = 0.9

# loading data
struct_log = '../data/HDFS/HDFS_100k.log_structured.csv'  # The structured log file
label_file = '../data/HDFS/anomaly_label.csv'  # The anomaly label file

(x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                            label_file=label_file,
                                                            window='session',
                                                            train_ratio=0.9,
                                                            split_type='uniform')

feature_extractor = preprocessing.FeatureExtractor()
x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
x_test = feature_extractor.transform(x_test)
aco = ACO()


def get_c_gamma(_x, _y):
    return \
        pow(2, C_VALUE_RANGE_STARTING_INDEX + _x), \
        pow(2, GAMMA_VALUE_RANGE_STARTING_INDEX + _y)


def get_svm_precision(c, gamma):
    # print(str(c) + ", " + str(gamma))
    model = SVM(C=c, gamma=gamma)
    model.fit(x_train, y_train)
    # print('Train validation:')
    # precision, recall, f1 = model.evaluate(x_train, y_train)

    # print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    return precision


def set_current_precision(_ant):
    x_index, y_index = _ant.get_position_x_y()
    precision = 0
    if aco.svm_training_result_matrix[x_index][y_index] == -1:
        c, gamma = get_c_gamma(x_index, y_index)
        precision = get_svm_precision(c, gamma)
        aco.svm_training_result_matrix[x_index][y_index] = precision
    else:
        # print("This precision is calculated already, no need to do it again.")
        precision = aco.svm_training_result_matrix[x_index][y_index]
    # also add this precision to the pheromone matrix - to update the pheromone level
    aco.global_pheromone_matrix[x_index][y_index] = aco.global_pheromone_matrix[x_index][y_index] + precision
    return precision


if __name__ == '__main__':
    # To record the time spent by ACO optimal C-gamma pair finding
    tic = time.time()

    aco.init_matrix()
    aco.init_pheromone_and_training_matrix()
    aco.init_ants()

    # set_current_precision(aco.ants[0])
    # print(aco.ants[0])
    # print(aco.svm_training_result_matrix)
    # a, b = aco.ants[0].move_ant_by_pheromone(aco.global_ant_position_matrix, aco.global_pheromone_matrix)
    # print(a)
    # print(b)
    #
    # print(aco.ants[0].visited)

    best_precision = 0
    best_x = 0
    best_y = 0

    for ant in aco.ants:
        current_precision = set_current_precision(ant)
        if current_precision > best_precision:
            best_precision = current_precision
            best_x = ant.current_position_x
            best_y = ant.current_position_y

    for index in range(0, MAX_ITERATION):
        for ant in aco.ants:
            # print(ant)
            next_x, next_y = ant.get_next_move_by_pheromone(aco.global_ant_position_matrix, aco.global_pheromone_matrix)
            # print(next_x, next_y)
            ant.move_to_next_position(next_x, next_y)

            current_precision = set_current_precision(ant)
            if current_precision > best_precision:
                best_precision = current_precision
                best_x = ant.current_position_x
                best_y = ant.current_position_y
        # print(aco.global_ant_position_matrix)
        aco.pheromone_evaporate()

    toc = time.time()

    print(f"Total execution time is: {toc - tic:0.4f} seconds.")

    print("The best precision achieved is: " + str(best_precision) + "." )
    print("Achieved by the optimal C-gamma pair: " + str(get_c_gamma(best_x, best_y)) + ".")
    print("The grid coordination for above optimal C-gamma pair is: " + str(best_x) + "," + str(best_y) + ".")

    print(aco.svm_training_result_matrix)

