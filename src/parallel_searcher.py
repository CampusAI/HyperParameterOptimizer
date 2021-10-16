
import time

class ParallelSearcher:
    def __init__(self, optimizer, job_class):
        """ Instantiate parallel searching

        Args:
            optimizer (GaussianProcessSearch): Optimizer used to find next points to test
            job_class (SearchJobInstance): Implementation of SearchJobInstance to manage jobs
        """
        self.optimizer = optimizer
        self.job_class = job_class

    def __launch(self, instance, candidate, retrials=5):
        launch_status = 1
        attempt = 0
        while launch_status != 0 and attempt < retrials:
            launch_status = instance.launch(**candidate)
            if launch_status != 0:
                print("There was some error launching the instance. Retrying (" + str(attempt) + "/" + retrials + ")")
                time.sleep(0.1)
                attempt += 1

    def optimize(self,
                 n_calls=10,
                 n_random_starts=5,
                 noise=0.01,
                 n_parallel_jobs=1,
                 refresh_rate=1,
                 first_id=0,
                 verbose=True,
                 plot_results=False):

        # Instantiate all initial jobs
        instances = [self.job_class(i) for i in range(first_id, first_id + n_parallel_jobs)]

        # Get all initial candidates
        candidates = []
        if len(self.optimizer.x_values) == 0:  # If first points, sample random
            candidates = self.optimizer.get_random_candidate(n_parallel_jobs)
        else:
            candidates = self.optimizer.get_next_candidate(n_parallel_jobs)

        # Launch all instances
        for i in range(n_parallel_jobs):
            print(candidates[i])
            self.__launch(instance=instances[i], candidate=candidates[i])
            n_calls -= 1

        while n_calls > 0:
            time.sleep(refresh_rate)  # refresh rate in seconds
            for i in range(n_parallel_jobs):
                instance = instances[i]
                if instance.done():
                    n_calls -= 1
                    instance_params = instance.passed_args
                    instance_result = instance.get_result()

                    # Display information
                    print("*****")
                    print("Finished job:", instance.id)
                    print("Instance_params:", instance_params)
                    print("Instance_result:", instance_result)
                    print("*****")

                    # Add point-evaluation info to the optimizer
                    self.optimizer.add_point_value(instance_params, instance_result)
                    self.optimizer.save_values()

                    # Instantiate new job instance
                    candidate = self.optimizer.get_next_candidate(1, n_random_starts)[0]
                    instances[i] = self.job_class(instance.id + n_parallel_jobs)
                    self.__launch(instance=instances[i], candidate=candidate)

                    # Display information
                    print("*****")
                    print("Starting job:", instances[i])
                    print("Instance_params:", candidate)
                    print("*****")