#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

template<typename T>
class Matrix {
 public:
  Matrix(int n, int m) : m(m), data(n * m) {}
  T &get(int i, int j) { return data[i * m + j]; }

 private:
  int m;
  std::vector<T> data;
};

enum MineState {
  UNKNOWN = -1,
  NO,
  YES,
};

class MinesweeperSolver {
 public:
  MinesweeperSolver(int n, int m) : n(n), m(m), clues(n, m), prob(n, m) {}
  void read_from_stdin() {
    std::string s;
    for (int j = 0; j < m; j++) std::cin >> s;
    for (int i = 0; i < n; i++) {
      std::cin >> s;
      for (int j = 0; j < m; j++) {
        std::cin >> s;
        if (s == "." || s == "F")
          clues.get(i, j) = -1;
        else
          clues.get(i, j) = std::stoi(s);
      }
    }
  }
  virtual void solve() = 0;

  int n, m;
  Matrix<int> clues;
  Matrix<double> prob;
};

class DFSSolver : public MinesweeperSolver {
 public:
  DFSSolver(int n, int m) : MinesweeperSolver(n, m), mines(n, m), tot(0) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) { mines.get(i, j) = MineState::UNKNOWN; }
  }
  bool clueless(int i, int j) {
    for (int di = -1; di <= 1; di++)
      for (int dj = -1; dj <= 1; dj++) {
        if (di != 0 || dj != 0) {
          if (!(0 <= i + di && i + di < n && 0 <= j + dj && j + dj < m)) continue;
          if (clues.get(i + di, j + dj) != -1) return false;
        }
      }
    return true;
  }
  bool feasible(int i, int j) {
    if (clues.get(i, j) == -1) return true;
    int min_mines = 0, max_mines = 0;
    for (int di = -1; di <= 1; di++)
      for (int dj = -1; dj <= 1; dj++)
        if (di != 0 || dj != 0) {
          if (!(0 <= i + di && i + di < n && 0 <= j + dj && j + dj < m)) continue;
          if (clues.get(i + di, j + dj) != -1) continue;
          switch (mines.get(i + di, j + dj)) {
            case MineState::UNKNOWN: max_mines += 1; break;
            case MineState::YES: min_mines += 1; max_mines += 1;
            default: break;
          }
        }
    return clues.get(i, j) >= min_mines && clues.get(i, j) <= max_mines;
  }
  bool check(int i, int j) {
    for (int di = -1; di <= 1; di++)
      for (int dj = -1; dj <= 1; dj++)
        if (di != 0 || dj != 0) {
          if (!(0 <= i + di && i + di < n && 0 <= j + dj && j + dj < m)) continue;
          if (!feasible(i + di, j + dj)) return false;
        }
    return true;
  }
  void dfs(int p) {
    if (p == n * m) {
      tot += 1;
      for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
          if (mines.get(i, j) == MineState::YES) prob.get(i, j) += 1.;
      return;
    }
    int i = p / m, j = p % m;
    if (clues.get(i, j) != -1 || clueless(i, j)) {
      dfs(p + 1);
      return;
    }
    const MineState choices[] = {MineState::NO, MineState::YES};
    for (auto state : choices) {
      mines.get(i, j) = state;
      if (check(i, j)) dfs(p + 1);
      mines.get(i, j) = MineState::UNKNOWN;
    }
  }
  void solve() override {
    dfs(0);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < m; j++) prob.get(i, j) /= tot;
  }

  Matrix<MineState> mines;
  int tot;
};

int main() {
  int n, m;
  std::cin >> n >> m;
  DFSSolver solver(n, m);
  solver.read_from_stdin();
  solver.solve();
  std::cout << std::fixed << std::setprecision(2);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++) {
      if (solver.clues.get(i, j) == -1) {
        if (solver.clueless(i, j))
          std::cout << "?";
        else
          std::cout << solver.prob.get(i, j);
      } else
        std::cout << "-";
      std::cout << (j == m - 1 ? "\n" : "\t");
    }
  return 0;
}