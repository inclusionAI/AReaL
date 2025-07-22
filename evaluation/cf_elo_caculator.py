import bisect
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm


def get_percentile(rating: float, sorted_ratings: List[float]) -> float:
    idx = bisect.bisect_left(sorted_ratings, float(rating))
    return round(idx / len(sorted_ratings) * 100, 1)


def read_ratings(file_path: str) -> List[float]:
    with open(file_path, "r") as f:
        ratings_dict = json.load(f)

    sorted_ratings = []
    for rating, count in ratings_dict.items():
        sorted_ratings.extend([float(rating)] * count)

    return sorted(sorted_ratings)


def load_cached_contest_data(cache_file_path: str) -> Dict:
    if not os.path.exists(cache_file_path):
        raise FileNotFoundError(f"Cache file does not exist: {cache_file_path}")

    with open(cache_file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} contest data from cache file")
    return data


def get_contest_data_from_cache(
    contest_id: int, cached_data: Dict
) -> Tuple[Optional[Dict], Optional[Dict]]:
    contest_id_str = str(contest_id)

    if contest_id_str not in cached_data:
        print(f"Warning: Contest {contest_id} data not found in cache")
        return None, None

    contest_data = cached_data[contest_id_str]

    try:
        standings = contest_data["standings"]
        rating_changes = contest_data["rating_changes"]

        if standings.get("status") != "OK" or rating_changes.get("status") != "OK":
            print(f"Warning: Contest {contest_id} cached data status abnormal")
            return None, None

        return standings, rating_changes

    except KeyError as e:
        print(f"Warning: Contest {contest_id} cached data structure abnormal: {e}")
        return None, None


def calc_elo_rating_offline(
    contest_id: int,
    problem_status: Dict[str, List[bool]],
    sorted_ratings: List[float],
    cached_data: Dict,
    pass_n=None,
) -> Optional[Tuple[int, float]]:
    try:
        standings, rating_changes = get_contest_data_from_cache(contest_id, cached_data)

        if standings is None or rating_changes is None:
            return None

        handle_set: Set[str] = set()
        try:
            handle_set_standings = set(
                standings["result"]["rows"][i]["party"]["members"][0]["handle"]
                for i in range(len(standings["result"]["rows"]))
            )

            handle_set_ratings = set(
                rating_changes["result"][i]["handle"]
                for i in range(len(rating_changes["result"]))
            )

            handle_set = handle_set_standings.intersection(handle_set_ratings)

            standings["result"]["rows"] = [
                row
                for row in standings["result"]["rows"]
                if row["party"]["members"][0]["handle"] in handle_set
            ]

            rating_changes["result"] = [
                change
                for change in rating_changes["result"]
                if change["handle"] in handle_set
            ]

            assert (
                len(standings["result"]["rows"]) == len(rating_changes["result"])
                and len(standings["result"]["rows"]) > 200
            )
        except Exception:
            return None

        if (
            "result" not in standings
            or "result" not in rating_changes
            or len(standings["result"]["rows"]) != len(rating_changes["result"])
            or len(standings["result"]["rows"]) <= 200
        ):
            return None

        max_rating = max(change["oldRating"] for change in rating_changes["result"])

        score = 0
        penalty = 0

        for problem in standings["result"]["problems"]:
            prob = f"{problem['contestId']}{problem['index']}"
            if prob in problem_status:
                if pass_n is None:
                    pass_n = len(problem_status[prob])
                for ith, status in enumerate(problem_status[prob][:pass_n]):
                    if status == 1.0:
                        if "points" in problem:
                            score += max(0, problem["points"] - 50 * ith)
                        else:
                            score += 1
                            penalty += ith * 10
                        break

        n = len(standings["result"]["rows"])

        rank = n
        for i in range(n):
            if standings["result"]["rows"][i]["points"] < score or (
                standings["result"]["rows"][i]["points"] == score
                and standings["result"]["rows"][i]["penalty"] > penalty
            ):
                rank = i
                break

        l, r = 0, max_rating + 100
        while r - l > 1:
            mid = (l + r) // 2
            new_seed = 1
            for i in range(n):
                new_seed += 1 / (
                    1 + 10 ** ((mid - rating_changes["result"][i]["oldRating"]) / 400)
                )
            if new_seed < rank:
                r = mid
            else:
                l = mid

        percentile = get_percentile(l, sorted_ratings)
        return l, percentile

    except Exception as e:
        print(f"Error calculating contest {contest_id} ELO rating: {e}")
        return None


def format_grouped_contest_data(
    submissions: List[List[bool]], problem_ids: List[str]
) -> List[Tuple[int, Dict[str, List[bool]]]]:
    if len(submissions) != len(problem_ids):
        raise ValueError("Length of submissions and problem_ids must be the same.")

    grouped_data = defaultdict(dict)

    for problem_id, submission in zip(problem_ids, submissions):
        # Extract contest ID using regex to capture leading digits
        match = re.match(r"(\d+)([A-Z].*)", problem_id)
        if not match:
            raise ValueError(f"Invalid problem ID format: {problem_id}")

        contest_id = int(match.group(1))
        problem_letter = match.group(0)

        grouped_data[contest_id][problem_letter] = submission

    combined_data = [
        (contest_id, problems) for contest_id, problems in grouped_data.items()
    ]

    return combined_data


def convert_score_to_cf_format(
    all_samples: List[Dict], metadata: List[str]
) -> List[List[bool]]:
    cf_results = []

    sorted_samples = sorted(all_samples, key=lambda x: x["idx"])

    for sample in sorted_samples:
        if "score" in sample:
            cf_results.append([bool(s) for s in sample["score"]])
        else:
            cf_results.append([False])

    return cf_results


class CFEloCalculator:
    def __init__(
        self,
        metadata_path: str = None,
        ratings_path: str = None,
        cache_file_path: str = None,
    ):
        # Set default paths
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = "/storage/openpsi/data/code/test_set/codeforces"

        self.metadata_path = metadata_path or os.path.join(
            current_dir, "metadata_cf.json"
        )
        self.ratings_path = ratings_path or os.path.join(
            current_dir, "ratings_2024.json"
        )
        self.cache_file_path = cache_file_path or os.path.join(
            current_dir, "all_contest_data.json"
        )

        # Preload data
        self._load_data()

    def _load_data(self):
        try:
            self.sorted_ratings = read_ratings(self.ratings_path)
            print(f"✓ Loaded {len(self.sorted_ratings)} historical rating data")

            with open(self.metadata_path, "r") as file:
                self.metadata = json.load(file)
            print(f"✓ Loaded {len(self.metadata)} problem metadata")

            self.cached_data = load_cached_contest_data(self.cache_file_path)
            print(f"✓ Loaded cached data")

        except Exception as e:
            raise RuntimeError(f"Failed to load data files: {e}")

    def calculate_elo(
        self, all_samples: List[Dict], pass_n: int = 1, verbose: bool = True
    ) -> Optional[Dict]:
        try:
            if verbose:
                print("\n" + "=" * 50)
                print("Starting Codeforces ELO rating calculation...")
                print("=" * 50)
            # Convert data format
            cf_results = convert_score_to_cf_format(all_samples, self.metadata)
            if verbose:
                print(f"✓ Converted {len(cf_results)} test results")

            # Format data
            model_results = format_grouped_contest_data(cf_results, self.metadata)
            if verbose:
                print(f"✓ Data grouped by {len(model_results)} contests")

            # Calculate ELO rating for each contest
            contest_elos = []
            skipped_contests = []

            iterator = (
                tqdm(model_results, desc="Calculating ELO ratings")
                if verbose
                else model_results
            )
            for contest_id, problems in iterator:
                elo_result = calc_elo_rating_offline(
                    contest_id, problems, self.sorted_ratings, self.cached_data, pass_n
                )
                if elo_result is not None:
                    contest_elos.append((contest_id, elo_result))
                else:
                    skipped_contests.append(contest_id)

            # Calculate average percentile
            percentiles = [elo[1][1] for elo in contest_elos if elo[1] is not None]
            ratings = [elo[1][0] for elo in contest_elos if elo[1] is not None]

            if not percentiles:
                print("Error: No valid percentiles calculated")
                return None

            estimated_rating = sum(ratings) / len(ratings)
            est_percentile = get_percentile(estimated_rating, self.sorted_ratings)

            # Display results
            if verbose:
                print("\n" + "=" * 50)
                print("CODEFORCES EVALUATION RESULTS")
                print("=" * 50)
                print(f"Estimated percentile: {est_percentile:.1f}%")
                print(f"Estimated Codeforces rating: {estimated_rating:.0f}")

                if skipped_contests:
                    print(
                        f"Skipped contest IDs: {skipped_contests[:10]}{'...' if len(skipped_contests) > 10 else ''}"
                    )

                print("=" * 50)

            # Return detailed results
            return {
                "estimated_percentile": est_percentile,
                "estimated_rating": estimated_rating,
                "contests_processed": len(contest_elos),
                "contests_skipped": len(skipped_contests),
                "skipped_contest_ids": skipped_contests,
                "individual_contest_results": [
                    {
                        "contest_id": contest_id,
                        "rating": rating,
                        "percentile": percentile,
                    }
                    for contest_id, (rating, percentile) in contest_elos
                ],
            }

        except Exception as e:
            print(f"Error calculating CF ELO rating: {e}")
            return None


def calculate_cf_elo_from_samples(
    all_samples: List[Dict],
    pass_n: int = 1,
    metadata_path: str = None,
    ratings_path: str = None,
    cache_file_path: str = None,
    verbose: bool = True,
) -> Optional[Dict]:

    calculator = CFEloCalculator(metadata_path, ratings_path, cache_file_path)
    return calculator.calculate_elo(all_samples, pass_n, verbose)
