import os
import shutil


def load_successes_from_disk(succ_dir, succ_traj, prune_trials, target_count,
                             cap_count=None, min_count=None):
    tuple_counts = {}
    for root, dirs, files in os.walk(succ_dir):
        for d in dirs:
            if d.count('-') == 4:
                goal, pickup, movable, receptacle, scene_num = d.split('-')
                # Add an entry for every successful trial folder in the directory.
                queue_for_delete = []
                deleted_all = True
                for _, _dirs, _ in os.walk(os.path.join(succ_dir, d)):
                    for _d in _dirs:
                        for _, _, _files in os.walk(os.path.join(succ_dir, d, _d)):
                            if 'video.mp4' in _files:
                                k = (goal, pickup, movable, receptacle, scene_num)
                                if k not in tuple_counts:
                                    tuple_counts[k] = 0
                                tuple_counts[k] += 1
                                deleted_all = False
                            else:
                                queue_for_delete.append(_d)
                            break  # only examine top level
                    break  # only examine top level
                if prune_trials:
                    if deleted_all:
                        print("Removing trial-less parent dir '%s'" % os.path.join(succ_dir, d))
                        shutil.rmtree(os.path.join(succ_dir, d))
                    else:
                        for _d in queue_for_delete:
                            print("Removing unfinished trial '%s'" % os.path.join(succ_dir, d, _d))
                            shutil.rmtree(os.path.join(succ_dir, d, _d))
        break  # only examine top level

    # Populate dataframe based on tuple constraints.
    for k in tuple_counts:
        if min_count is None or tuple_counts[k] >= min_count:
            to_add = tuple_counts[k] if cap_count is None else cap_count
            for _ in range(to_add):
                succ_traj = succ_traj.append({
                    "goal": k[0],
                    "pickup": k[1],
                    "movable": k[2],
                    "receptacle": k[3],
                    "scene": k[4]}, ignore_index=True)
    tuples_at_target_count = set([t for t in tuple_counts if tuple_counts[t] >= target_count])

    return succ_traj, tuples_at_target_count


def load_fails_from_disk(succ_dir, to_write=None):
    fail_traj = set()
    fail_dir = os.path.join(succ_dir, "fails")
    if not os.path.isdir(fail_dir):
        os.makedirs(fail_dir)
    if to_write is not None:
        for goal, pickup, movable, receptacle, scene_num in to_write:
            with open(os.path.join(fail_dir, '-'.join([goal, pickup, movable, receptacle, scene_num])), 'w') as f:
                f.write("0")
    for root, dirs, files in os.walk(fail_dir):
        for fn in files:
            if fn.count('-') == 4:
                goal, pickup, movable, receptacle, scene_num = fn.split('-')
                fail_traj.add((goal, pickup, movable, receptacle, scene_num))
        break  # only examine top level
    return fail_traj
