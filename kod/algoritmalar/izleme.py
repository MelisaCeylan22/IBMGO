"""Aramanın gerçek amaç fonksiyonu çağrılarını, rastgele akışı değiştirmeden izler."""
import numpy as np


class EvaluationTrace:
    def __init__(self, target):
        self.target = target
        self.events = []
        self.best = float("inf")
        self.initial = None

    def record(self, score, step, phase):
        if self.target is None:
            return
        self.best = min(self.best, float(score))
        self.events.append(dict(evaluation=len(self.events) + 1, step=step,
                                phase=phase, score=float(score), best=self.best))

    def finish_initial(self, score):
        self.initial = float(score)

    def finish(self, final, steps):
        if self.target is None:
            return
        first = next(e for e in self.events
                     if np.isclose(e["best"], final, rtol=1e-10, atol=1e-12))
        improvements = [self.events[0]]
        for previous, event in zip(self.events, self.events[1:]):
            if event["best"] < previous["best"] and not np.isclose(
                    event["best"], previous["best"], rtol=1e-10, atol=1e-12):
                improvements.append(event)
        last = improvements[-1]
        self.target.clear()
        self.target.update(events=self.events, initial_fitness=self.initial,
                           final_fitness=float(final), steps=steps,
                           evaluations=len(self.events), first_final_step=first["step"],
                           first_final_evaluation=first["evaluation"],
                           last_improvement_step=last["step"],
                           last_improvement_evaluation=last["evaluation"],
                           last_improvement_phase=last["phase"])
