"""
Reconstructed Cohort Method (UNESCO) — Python implementation
==============================================================

INPUTS (per grade, e.g. Grade I..V):
  - enrollment_prev_year : enrollment by grade in year (t-1)
  - enrollment_curr_year : enrollment by grade in year t
  - repeaters_curr_year  : repeaters by grade in year t
  - passed_final_grade   : number of students who PASSED the final grade
                            (i.e. graduated) in year t

METHOD:
  1. Promotion (P), Repetition (R) and Dropout (D) rates are derived per
     grade from two consecutive years of enrollment + repeaters + the
     number who passed the final grade.
  2. A hypothetical cohort (default 1000 pupils) entering Grade I is
     simulated forward year-by-year, splitting into "promoted",
     "repeated" and "dropped-out" streams using the rates above, until the residual
     mass still moving through the system is negligible.
  3. Standard system-efficiency indicators are computed from the
     simulated flows: survival rate, dropout rate, output (graduates),
     total student-years, average years per graduate, coefficient of
     efficiency, etc.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
@dataclass
class CohortResult:
    grade_labels: List[str]
    promotion_rate: List[float]
    repetition_rate: List[float]
    dropout_rate: List[float]
    survival_rate_by_grade: List[float]   # index g = share of cohort ever reaching grade g
    total_output: float                   # graduates produced by the cohort
    total_dropouts: float
    total_repeaters: float
    total_student_years: float
    years_per_graduate: float
    coefficient_of_efficiency: float
    cohort_dropout_rate: float            # share of the ORIGINAL cohort lost to dropout
    cohort_size: float
    ideal_years: int
    years_simulated: int

    def summary(self) -> str:
        lines = []
        lines.append("RECONSTRUCTED COHORT — SYSTEM EFFICIENCY INDICATORS")
        lines.append("=" * 55)
        lines.append(f"{'Grade':<10}{'Promotion':>12}{'Repetition':>12}{'Dropout':>12}{'Survival→':>12}")
        for i, g in enumerate(self.grade_labels):
            lines.append(
                f"{g:<10}{self.promotion_rate[i]*100:>11.2f}%"
                f"{self.repetition_rate[i]*100:>11.2f}%"
                f"{self.dropout_rate[i]*100:>11.2f}%"
                f"{self.survival_rate_by_grade[i]*100:>11.2f}%"
            )
        lines.append("-" * 55)
        lines.append(f"Cohort size (hypothetical)      : {self.cohort_size:,.0f}")
        lines.append(f"Total output (graduates)         : {self.total_output:,.2f}")
        lines.append(f"Total dropouts                   : {self.total_dropouts:,.2f}")
        lines.append(f"Total repeaters (repeat-episodes) : {self.total_repeaters:,.2f}")
        lines.append(f"Total student-years used          : {self.total_student_years:,.2f}")
        lines.append(f"Years input per graduate           : {self.years_per_graduate:.3f}  (ideal = {self.ideal_years})")
        lines.append(f"Coefficient of efficiency          : {self.coefficient_of_efficiency*100:.2f}%")
        lines.append(f"Cohort dropout rate                 : {self.cohort_dropout_rate*100:.2f}%")
        lines.append(f"Survival rate to final grade        : {self.survival_rate_by_grade[-1]*100:.2f}%")
        return "\n".join(lines)


def compute_rates(enrollment_prev_year: List[float],
                   enrollment_curr_year: List[float],
                   repeaters_curr_year: List[float],
                   passed_final_grade: float):
    """
    Derive promotion / repetition / dropout rates per grade.

    For grades 1..G-1:
        P_g = (E_{g+1,t} - Repeaters_{g+1,t}) / E_{g,t-1}
        R_g = Repeaters_{g,t} / E_{g,t-1}
        D_g = 1 - P_g - R_g

    For the final grade G (no "next grade" enrollment to compare against):
        P_G = Passed_t / E_{G,t-1}
        R_G = Repeaters_{G,t} / E_{G,t-1}
        D_G = 1 - P_G - R_G
    """
    G = len(enrollment_prev_year)
    assert len(enrollment_curr_year) == G
    assert len(repeaters_curr_year) == G

    P = [0.0] * G
    R = [0.0] * G
    D = [0.0] * G

    for g in range(G - 1):
        P[g] = (enrollment_curr_year[g + 1] - repeaters_curr_year[g + 1]) / enrollment_prev_year[g]
        R[g] = repeaters_curr_year[g] / enrollment_prev_year[g]
        D[g] = 1 - P[g] - R[g]

    P[G - 1] = passed_final_grade / enrollment_prev_year[G - 1]
    R[G - 1] = repeaters_curr_year[G - 1] / enrollment_prev_year[G - 1]
    D[G - 1] = 1 - P[G - 1] - R[G - 1]

    return P, R, D


def run_reconstructed_cohort(enrollment_prev_year: List[float],
                              enrollment_curr_year: List[float],
                              repeaters_curr_year: List[float],
                              passed_final_grade: float,
                              grade_labels: List[str] = None,
                              cohort_size: float = 1000,
                              ideal_years: int = None,
                              max_years: int = 60,
                              convergence_threshold: float = 1e-6) -> CohortResult:
    """
    Run the full Reconstructed Cohort Method simulation and return all
    system-efficiency indicators.

    Parameters
    ----------
    enrollment_prev_year : enrollment by grade, earlier year (t-1)
    enrollment_curr_year : enrollment by grade, later year (t)
    repeaters_curr_year  : repeaters by grade, later year (t)
    passed_final_grade   : number of students who passed/graduated the
                            final grade in year t
    grade_labels         : optional names, e.g. ["Grade I", ..., "Grade V"]
    cohort_size           : size of the hypothetical starting cohort (default 1000)
    ideal_years           : "ideal" number of years to complete the cycle
                            (defaults to the number of grades)
    max_years              : simulation horizon (safety cap)
    convergence_threshold : stop early once remaining active mass is smaller
                             than this fraction of the cohort
    """
    G = len(enrollment_prev_year)
    if grade_labels is None:
        grade_labels = [f"Grade {i+1}" for i in range(G)]
    if ideal_years is None:
        ideal_years = G

    P, R, D = compute_rates(enrollment_prev_year, enrollment_curr_year,
                             repeaters_curr_year, passed_final_grade)

    # new_entrants[y][g]: students entering grade g for the FIRST time in year y
    # repeat_mass[y][g]:  students repeating grade g in year y
    new_entrants = [[0.0] * G for _ in range(max_years + 1)]
    repeat_mass = [[0.0] * G for _ in range(max_years + 1)]
    dropouts = [[0.0] * G for _ in range(max_years + 1)]
    graduates_by_year = [0.0] * (max_years + 1)

    new_entrants[0][0] = cohort_size

    years_used = max_years
    for y in range(max_years):
        active_mass = sum(new_entrants[y][g] + repeat_mass[y][g] for g in range(G))
        if active_mass < cohort_size * convergence_threshold:
            years_used = y
            break
        for g in range(G):
            total_g = new_entrants[y][g] + repeat_mass[y][g]
            if total_g == 0:
                continue
            promoted = total_g * P[g]
            repeated = total_g * R[g]
            dropped = total_g * D[g]
            dropouts[y][g] += dropped
            if g < G - 1:
                new_entrants[y + 1][g + 1] += promoted
            else:
                graduates_by_year[y + 1] += promoted
            repeat_mass[y + 1][g] += repeated

    total_output = sum(graduates_by_year)
    total_dropouts = sum(sum(row) for row in dropouts)
    total_repeaters = sum(sum(repeat_mass[y]) for y in range(1, max_years + 1))
    total_student_years = sum(
        new_entrants[y][g] + repeat_mass[y][g]
        for y in range(max_years + 1) for g in range(G)
    )

    # survival: share of the cohort that ever enters each grade for the first time
    survival_rate_by_grade = []
    for g in range(G):
        entrants_g = sum(new_entrants[y][g] for y in range(max_years + 1))
        survival_rate_by_grade.append(entrants_g / cohort_size)

    years_per_graduate = total_student_years / total_output if total_output else float("nan")
    coefficient_of_efficiency = (ideal_years * total_output) / total_student_years if total_student_years else float("nan")
    cohort_dropout_rate = total_dropouts / cohort_size

    return CohortResult(
        grade_labels=grade_labels,
        promotion_rate=P,
        repetition_rate=R,
        dropout_rate=D,
        survival_rate_by_grade=survival_rate_by_grade,
        total_output=total_output,
        total_dropouts=total_dropouts,
        total_repeaters=total_repeaters,
        total_student_years=total_student_years,
        years_per_graduate=years_per_graduate,
        coefficient_of_efficiency=coefficient_of_efficiency,
        cohort_dropout_rate=cohort_dropout_rate,
        cohort_size=cohort_size,
        ideal_years=ideal_years,
        years_simulated=years_used,
    )


def get_inputs_interactively():
    """Prompt the user on the command line for enrollment / repeaters / passed data."""
    n = int(input("How many grades in the cycle? (e.g. 5 for Grade I-V): ").strip())
    grade_labels = []
    for i in range(n):
        label = input(f"  Label for grade {i+1} [default 'Grade {i+1}']: ").strip()
        grade_labels.append(label if label else f"Grade {i+1}")

    print("\nEnter ENROLLMENT for the EARLIER year (t-1), grade by grade:")
    enrollment_prev_year = [float(input(f"  {g}: ")) for g in grade_labels]

    print("\nEnter ENROLLMENT for the LATER year (t), grade by grade:")
    enrollment_curr_year = [float(input(f"  {g}: ")) for g in grade_labels]

    print("\nEnter REPEATERS for the LATER year (t), grade by grade:")
    repeaters_curr_year = [float(input(f"  {g}: ")) for g in grade_labels]

    passed_final_grade = float(input(f"\nHow many students PASSED the final grade "
                                      f"({grade_labels[-1]}) in the later year? "))

    cohort_size = input("\nHypothetical cohort size to simulate [default 1000]: ").strip()
    cohort_size = float(cohort_size) if cohort_size else 1000

    return dict(
        enrollment_prev_year=enrollment_prev_year,
        enrollment_curr_year=enrollment_curr_year,
        repeaters_curr_year=repeaters_curr_year,
        passed_final_grade=passed_final_grade,
        grade_labels=grade_labels,
        cohort_size=cohort_size,
    )


if __name__ == "__main__":
    import sys

    if "--interactive" in sys.argv:
        inputs = get_inputs_interactively()
        result = run_reconstructed_cohort(**inputs)
        print()
        print(result.summary())
    else:
        # Demo / validation: reproduces the numbers in
        # Reconstructed_Cohort_Method_cohort
        # primary cycle Grade I-V, 2024 -> 2025).
        grade_labels = ["Grade I", "Grade II", "Grade III", "Grade IV", "Grade V"]
        enrollment_2024 = [1618074,	1598247,	1550106,	1472547,	1364560]
        enrollment_2025 = [1555274,	1545743,	1520342,	1452134,	1359115]
        repeaters_2025 = [99870,	51532,	49477,	37672,	24018]
        passed_2025 = 1327542  # number who passed/graduated Grade V in 2025

        result = run_reconstructed_cohort(
            enrollment_prev_year=enrollment_2024,
            enrollment_curr_year=enrollment_2025,
            repeaters_curr_year=repeaters_2025,
            passed_final_grade=passed_2025,
            grade_labels=grade_labels,
        )

        print("(Demo using the numbers from the given data. Run with --interactive")
        print(" to type in your own enrollment / repeaters / passed figures.)\n")
        print(result.summary())