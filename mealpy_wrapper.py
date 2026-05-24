import numpy as np
from mealpy.optimizer import Optimizer
from scipy import special

class GoatMuleClimbingOptimization(Optimizer):
    """
    Goat-Mule Climbing Optimization (GMCO) coded as a native Mealpy Extension.
    Combines Goat's Lévy-driven exploration with Mule's tangent-constrained 
    Givens hyperplane exploitation.
    """
    def __init__(self, epoch=150, pop_size=40, theta_max=15.0, w_goat=0.3, w_mule=0.2, **kwargs):
        super().__init__(**kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.theta_max = np.radians(theta_max)
        self.w_goat = w_goat
        self.w_mule = w_mule
        self.zigzag_direction = 1.0

    def _levy_flight(self, dim, Lambda=1.5):
        num = special.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)
        den = special.gamma((1 + Lambda) / 2) * Lambda * (2 ** ((Lambda - 1) / 2))
        sigma = (num / den) ** (1 / Lambda)
        u = np.random.normal(0, sigma, size=dim)
        v = np.random.normal(0, 1, size=dim)
        return u / (np.abs(v) ** (1 / Lambda))

    def _estimate_topology(self, pos):
        eps = 1e-5
        dim = len(pos)
        grad = np.zeros(dim)
        base_fit = self.problem.get_fitness_position(pos)
        
        for i in range(dim):
            perturbed = np.copy(pos)
            perturbed[i] += eps
            grad[i] = (self.problem.get_fitness_position(perturbed) - base_fit) / eps
            
        roughness = np.clip(np.std(grad) / (np.mean(np.abs(grad)) + 1e-6), 0.1, 1.0)
        return grad, roughness

    def evolve(self, epoch):
        half_pop = self.pop_size // 2
        g_best_pos = self.g_best.solution
        
        for idx in range(self.pop_size):
            agent_pos = np.copy(self.pop[idx].solution)
            dim = len(agent_pos)
            grad, rough = self._estimate_topology(agent_pos)
            grad_norm = np.linalg.norm(grad)
            
            # 1. KEÇİ MODU (GLOBAL KEŞİF)
            if idx < half_pop:
                dir_vector = grad / grad_norm if grad_norm > 0 else np.zeros(dim)
                social_dir = g_best_pos - agent_pos
                sn = np.linalg.norm(social_dir)
                if sn > 0: social_dir /= sn
                
                jump = self._levy_flight(dim) * rough * 0.4
                new_pos = agent_pos + (self.w_goat * dir_vector) + (0.2 * np.random.rand() * social_dir) + jump
            
            # 2. KATIR MODU (HASSAS SÖMÜRÜ / GIVENS ROTASYONU)
            else:
                if grad_norm == 0:
                    continue
                actual_slope_angle = np.arctan(grad_norm)
                mule_dir = grad / grad_norm
                
                if actual_slope_angle > self.theta_max:
                    cos_phi = np.tan(self.theta_max) / np.tan(actual_slope_angle)
                    cos_phi = np.clip(cos_phi, -1.0, 1.0)
                    phi = np.arccos(cos_phi) * self.zigzag_direction
                    
                    if dim >= 2:
                        axis_idx = np.random.choice(dim, 2, replace=False)
                        i, j = axis_idx[0], axis_idx[1]
                        c, s = np.cos(phi), np.sin(phi)
                        vi_new = c * mule_dir[i] - s * mule_dir[j]
                        vj_new = s * mule_dir[i] + c * mule_dir[j]
                        mule_dir[i], mule_dir[j] = vi_new, vj_new
                        
                    if np.random.rand() < 0.1 + (rough * 0.2):
                        self.zigzag_direction *= -1
                
                social_dir = g_best_pos - agent_pos
                sn = np.linalg.norm(social_dir)
                if sn > 0: social_dir /= sn
                
                new_pos = agent_pos + (self.w_mule * (1.0 - rough * 0.5) * mule_dir) + (0.2 * np.random.rand() * social_dir)
            
            new_pos = self.problem.clip_to_bounds(new_pos)
            self.pop[idx].solution = new_pos
