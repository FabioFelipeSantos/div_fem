import math
from re import L
from typing import Any, cast, Dict

import numpy as np
from numpy.typing import NDArray


class Mesh2DDegreeOfFreedomOptimization:

    elements_points: Dict[int, list[int]]
    elements_dof: Dict[int, Dict[int, list[int]]]
    nip: int
    next_point_number: int

    received_max_min_points_number: list[int]
    dof_number: int
    n_elements: int
    total_dof_number: int

    extracted_dof_per_element: Dict[int, list[int]]

    def __init__(self, elements: list[tuple[int, int]], number_of_interpolation_points: int, number_of_dof: int) -> None:
        self.n_elements = len(elements)
        self.nip = number_of_interpolation_points
        self.dof_number = number_of_dof

        self.received_max_min_points_number = self._received_points_number(elements)
        self.next_point_number = self.received_max_min_points_number[1] + 1
        self._calculating_additional_interpolation_points_number(elements)

        self.number_of_points = self.next_point_number - 1
        self.total_dof_number = self.number_of_points * self.dof_number

    def _calculating_additional_interpolation_points_number(self, elements: list[tuple[int, int]]) -> None:
        self.elements_points = {}
        for i, element in enumerate(elements):
            self.elements_points.update({i + 1: [*element, *[self.next_point_number + value for value in range(self.nip - 2)]]})
            self.next_point_number += self.nip - 2

    def _received_points_number(self, elements: list[tuple[int, int]]) -> list[int]:
        min_max_points_number = [math.inf, -math.inf]

        for initial_point, final_point in elements:
            if initial_point < min_max_points_number[0]:
                min_max_points_number[0] = initial_point

            if final_point > min_max_points_number[1]:
                min_max_points_number[1] = final_point

        return [int(value) for value in min_max_points_number]

    def _extracting_dof_from_element(self) -> Dict[int, list[int]]:
        extracted_dof_per_element = {}

        for idx, element in self.elements_dof.items():
            nodes_dof: list[int] = []

            for node_dof in element.values():
                nodes_dof = [*nodes_dof, *node_dof]

            extracted_dof_per_element.update({idx: nodes_dof})

        return extracted_dof_per_element

    def _evaluate_individual(self, chromosome: np.ndarray) -> int:
        """
        Calcula a banda MÁXIMA da malha inteira para um determinado cromossomo.
        O cromossomo é uma lista contendo a ordem global dos DOFs.
        """
        max_mesh_bandwidth = 0

        # Mapeando os DOFs globais para os nós baseados no cromossomo
        dof_per_node = {
            index + 1: chromosome[index * self.dof_number : (index + 1) * self.dof_number].tolist() for index in range(self.number_of_points)
        }

        # Verificando cada elemento
        for element_nodes in self.elements_points.values():
            element_dofs = []
            for node in element_nodes:
                element_dofs.extend(dof_per_node[node])

            # Banda deste elemento específico
            element_bandwidth = max(element_dofs) - min(element_dofs)

            # Atualiza a banda máxima da malha se este elemento for pior
            if element_bandwidth > max_mesh_bandwidth:
                max_mesh_bandwidth = element_bandwidth

        return max_mesh_bandwidth

    def _crossover_order(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        Crossover de Ordem (Order Crossover - OX1).
        Garante que os DOFs não se repitam e nenhum seja perdido.
        """
        size = len(parent1)
        start, end = sorted(np.random.choice(size, 2, replace=False))

        child = np.full(size, -1)
        # 1. Copia um trecho do Parente 1 para o Filho
        child[start:end] = parent1[start:end]

        # 2. Preenche o restante com a ordem do Parente 2, ignorando o que já existe
        p2_filtered = [gene for gene in parent2 if gene not in child]

        # Insere os valores faltantes
        pointer = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = p2_filtered[pointer]
                pointer += 1

        return child

    def _mutate_swap(self, chromosome: np.ndarray) -> np.ndarray:
        """Mutação por troca: escolhe duas posições aleatórias e inverte os valores."""
        idx1, idx2 = np.random.choice(len(chromosome), 2, replace=False)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def optimization(self) -> None:
        print("\n\nIniciando Otimização da Malha...")

        # Parâmetros do GA
        pop_size = 50
        max_generations = 100
        mutation_probability = 0.3

        # 1. Criando a População Inicial (50 malhas aleatórias)
        # Cada indivíduo é uma permutação de 0 até (Total de DOFs - 1)
        population = [np.random.permutation(self.total_dof_number) for _ in range(pop_size)]

        best_overall_bandwidth = math.inf
        best_chromosome = None

        for generation in range(max_generations):
            # 2. Avaliação de toda a população
            # O fitness aqui usa aquela nossa lógica: (MaxPossivel - Banda) + 1
            max_possible_bandwidth = self.total_dof_number - 1
            fitness_scores = []
            bandwidths = []

            for ind in population:
                bw = self._evaluate_individual(ind)
                bandwidths.append(bw)
                fitness = (max_possible_bandwidth - bw) + 1
                fitness_scores.append(fitness)

                # Salvando o melhor de todos os tempos
                if bw < best_overall_bandwidth:
                    best_overall_bandwidth = bw
                    best_chromosome = ind.copy()

            print(f"Geração {generation:3d} | Melhor Banda Atual: {min(bandwidths)} | Melhor Global: {best_overall_bandwidth}")

            # 3. Seleção (Roleta baseada no fitness)
            total_fitness = sum(fitness_scores)
            probabilities = [f / total_fitness for f in fitness_scores]

            new_population = []

            # Elitismo: Salvar o melhor indivíduo para a próxima geração sem alterações
            best_idx = np.argmin(bandwidths)
            new_population.append(population[best_idx].copy())

            # 4. Reprodução (Crossover e Mutação)
            while len(new_population) < pop_size:
                # Seleciona pais pela roleta
                p1_idx = np.random.choice(pop_size, p=probabilities)
                p2_idx = np.random.choice(pop_size, p=probabilities)
                parent1, parent2 = population[p1_idx], population[p2_idx]

                # Crossover
                child = self._crossover_order(parent1, parent2)

                # Mutação
                if np.random.rand() < mutation_probability:
                    child = self._mutate_swap(child)

                new_population.append(child)

            population = new_population

        print("\nOtimização Concluída!")
        print(f"Banda Otimizada (Mínima): {best_overall_bandwidth}")
