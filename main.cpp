#include <iostream>
#include <array>
#include <cmath>

template <size_t V>
class VPTree {
  public:
    using Point = std::array<double, V>; // point as array, e.g in R^2, (x,y)
    using Metric = double(*)(const Point&, const Point&);
  private:
    struct Node {
      Point p;
      double mu;
      Node* left, right;
    }
    Node(const Point& vantage_point, double median) 
        : p(vantage_point), mu(median), left(nullptr), right(nullptr) {}

    Node* root = nullptr;
    Metric d = float absolute(int x, int y) {
      return std::abs(x - y);
    }

    Node build_tree(std::vector<Point>& S) {
      if (S.empty()) {
        return NULL; // todo: lookif nullptr should be here           
      }

    
    return nullptr;
    }

    Point select_vp(std::vector<Point>& S) {
      size_t sample_size = std::min<size_t>(100, size_t S.size());
      std::vector<Point>& random_sample;
      std::sample(S.begin(), S.end(), std::back_inserter(random_sample), sample_size,
                  std::mt19937 {std::random_device{}()}); // Take random sample of set S
      double best_spread = 0;

      


    }
};

