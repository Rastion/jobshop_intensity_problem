from qubots.base_problem import BaseProblem
import random
import os

class JobShopIntensityProblem(BaseProblem):
    """
    Job Shop Scheduling with Intensity Problem for Qubots.
    
    In this problem, each job consists of an ordered sequence of tasks (activities), one per machine.
    A task’s progress is driven by the intensity of the machine at each time step. In particular,
    an activity starting at time t is considered complete once the sum of the machine’s intensity
    from time t onward meets or exceeds its processing time. Each machine processes tasks one at a time,
    and for each machine a candidate solution specifies a permutation (order) of the jobs.
    
    **Candidate Solution Representation:**
      A dictionary with key "jobs_order" mapping to a list of length nb_machines.
      Each element is a permutation (list) of job indices (0-indexed) representing the sequence
      in which jobs are processed on that machine.
    """
    
    def __init__(self, instance_file: str):
        (self.nb_jobs,
         self.nb_machines,
         self.time_horizon,
         self.processing_time,
         self.machine_order,
         self.intensity) = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = f.readlines()
        # Remove empty lines.
        lines = [line.strip() for line in lines if line.strip()]
        
        # Second line contains: nb_jobs, nb_machines, time_horizon.
        first_line = lines[1].split()
        nb_jobs = int(first_line[0])
        nb_machines = int(first_line[1])
        time_horizon = int(first_line[2])
        
        # Processing times (in processing order) for each job:
        processing_times_in_order = []
        for i in range(3, 3 + nb_jobs):
            parts = lines[i].split()
            times = [int(x) for x in parts[:nb_machines]]
            processing_times_in_order.append(times)
        
        # Processing order of machines for each job:
        machine_order = []
        for i in range(4 + nb_jobs, 4 + 2 * nb_jobs):
            parts = lines[i].split()
            order = [int(x) - 1 for x in parts[:nb_machines]]
            machine_order.append(order)
        
        # Reorder processing times: For each job j and machine m, 
        # processing_time[j][m] is the processing time of job j's task that is processed on machine m.
        processing_time = []
        for j in range(nb_jobs):
            proc = []
            for m in range(nb_machines):
                pos = machine_order[j].index(m)
                proc.append(processing_times_in_order[j][pos])
            processing_time.append(proc)
        
        # Intensity for each machine (each line has time_horizon integers).
        intensity = []
        start_index = 5 + 2 * nb_jobs
        for i in range(start_index, start_index + nb_machines):
            parts = lines[i].split()
            intens = [int(x) for x in parts[:time_horizon]]
            intensity.append(intens)
        
        return nb_jobs, nb_machines, time_horizon, processing_time, machine_order, intensity
    
    def evaluate_solution(self, solution) -> float:
        penalty = 1e9
        # Candidate solution must be a dict with key "jobs_order".
        if not isinstance(solution, dict) or "jobs_order" not in solution:
            return penalty
        candidate = solution["jobs_order"]
        if not isinstance(candidate, list) or len(candidate) != self.nb_machines:
            return penalty
        for m in range(self.nb_machines):
            if sorted(candidate[m]) != list(range(self.nb_jobs)):
                return penalty
        
        # For simulation purposes, we compute finish times for each task.
        # finish[j][m] will store the finish time for job j's task on machine m.
        finish = [[0] * self.nb_machines for _ in range(self.nb_jobs)]
        
        # For each job j, determine the position of each machine in its processing order.
        # job_machine_position[j][m] = position of machine m in machine_order[j]
        job_machine_position = [[-1]*self.nb_machines for _ in range(self.nb_jobs)]
        for j in range(self.nb_jobs):
            for pos, m in enumerate(self.machine_order[j]):
                job_machine_position[j][m] = pos
        
        # Define a helper: given a start time s on machine m and required processing p,
        # compute the earliest finish time f (s ≤ f ≤ time_horizon) such that
        # sum_{t=s}^{f-1} intensity[m][t] >= p.
        def compute_finish_time(s, m, p):
            total = 0
            t = s
            while t < self.time_horizon and total < p:
                total += self.intensity[m][t]
                t += 1
            if total < p:
                # Infeasible: not enough intensity.
                return self.time_horizon + 1000
            return t
        
        # For each machine m, process tasks in the candidate order (for that machine).
        for m in range(self.nb_machines):
            seq = candidate[m]
            for idx, j in enumerate(seq):
                # For job j on machine m, its earliest start time is the maximum of:
                # - finish time of the previous task in job j (if any)
                # - finish time of the previous job on machine m (if any)
                prev_job_finish = 0
                if idx > 0:
                    prev_j = seq[idx - 1]
                    prev_job_finish = finish[prev_j][m]
                prev_machine_finish = 0
                pos = job_machine_position[j][m]
                if pos > 0:
                    # The previous machine in job j's order:
                    prev_m = self.machine_order[j][pos - 1]
                    prev_machine_finish = finish[j][prev_m]
                s = max(prev_job_finish, prev_machine_finish)
                f_time = compute_finish_time(s, m, self.processing_time[j][m])
                finish[j][m] = f_time
        
        # The makespan is the maximum finish time among the last tasks for each job.
        job_completion = []
        for j in range(self.nb_jobs):
            m_last = self.machine_order[j][-1]
            job_completion.append(finish[j][m_last])
        makespan = max(job_completion)
        return makespan
    
    def random_solution(self):
        candidate = []
        for m in range(self.nb_machines):
            perm = list(range(self.nb_jobs))
            random.shuffle(perm)
            candidate.append(perm)
        return {"jobs_order": candidate}
