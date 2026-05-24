import numpy as np
from mealpy import Problem
from mealpy_wrapper import GoatMuleClimbingOptimization
from mealpy.swarm_based.ACO import OriginalACO # Karşılaştırma için Mealpy ACO motoru

# --- TSPLIB OLIVER30 KOORDİNATLARI (Küresel Standart Veri Seti) ---
OLIVER30_COORDS = np.array([
    [54, 67], [54, 62], [37, 84], [41, 94], [2, 99], [7, 64], [25, 62], [22, 60], [18, 54], [4, 50],
    [13, 40], [18, 40], [24, 42], [25, 38], [44, 35], [41, 26], [45, 21], [58, 35], [62, 32], [82, 7],
    [91, 38], [83, 46], [71, 44], [64, 60], [68, 58], [83, 69], [87, 76], [74, 78], [71, 71], [58, 69]
])

def compute_tsplib_distance_matrix(coords):
    """ Şehirler arası Öklid mesafesini hesaplayan maliyet matrisi """
    num_cities = len(coords)
    matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    return matrix

DIST_MATRIX = compute_tsplib_distance_matrix(OLIVER30_COORDS)

def tsplib_objective_function(solution):
    """
    Sürekli uzaydaki koordinatları kesikli permütasyona (şehir sırasına)
    dönüştüren kararlı sıralama (Discrete Mapping) ve Toplam Yol Hesabı
    """
    ordered_cities = np.argsort(solution)
    total_distance = 0.0
    for i in range(len(ordered_cities) - 1):
        total_distance += DIST_MATRIX[ordered_cities[i], ordered_cities[i+1]]
    # Turun tamamlanması (Son şehirden ilk şehre dönüş)
    total_distance += DIST_MATRIX[ordered_cities[-1], ordered_cities[0]]
    return float(total_distance)

def execute_tsplib_comparison():
    print("==================================================================")
    # 2026 yılı güncel standartlarında NP-Hard Kombinatoryal Karşılaştırma Motoru
    print("    GMCO vs. ACO: TSPLIB NP-HARD COMBINATORIAL EVALUATION ENGINE  ")
    print("==================================================================")
    
    num_cities = len(OLIVER30_COORDS)
    epochs = 150
    pop_size = 40
    
    # Mealpy üzerinde TSP probleminin soyutlanması
    tsp_problem = Problem(
        bounds=[-5.0, 5.0],
        minmax="min",
        name="TSPLIB_Oliver30",
        obj_func=tsplib_objective_function
    )
    
    # 1. Geliştirdiğimiz GMCO Motorunun Çalıştırılması
    print("[RUNNING] Executing Discrete GMCO on Oliver30 Matrix...")
    gmco_model = GoatMuleClimbingOptimization(epoch=epochs, pop_size=pop_size, theta_max=15.0)
    gmco_best = gmco_model.solve(tsp_problem)
    
    # 2. Karınca Kolonisi (ACO) Motorunun Çalıştırılması
    print("[RUNNING] Executing Original ACO on Oliver30 Matrix...")
    aco_model = OriginalACO(epoch=epochs, pop_size=pop_size)
    aco_best = aco_model.solve(tsp_problem)
    
    print("\n=================== NİHAİ BAŞARI ANALİZİ ===================")
    print(f"TSPLIB Oliver30 En İyi Bilinen Rota Uzunluğu (Optimum): 420.0")
    print(f"-> DISCRETE GMCO En İyi Rota Uzunluğu: {gmco_best.target.fitness:.2f}")
    print(f"-> STANDARD ACO En İyi Rota Uzunluğu:  {aco_best.target.fitness:.2f}")
    print("============================================================\n")

if __name__ == '__main__':
    execute_tsplib_comparison()
