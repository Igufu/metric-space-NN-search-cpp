#include <iostream>
#include <array>
#include <cmath>

template <size_t V>
class VPTree {
  public:
    using Point = std::array<double, V>; // point as array, e.g in R^2, (x,y)
    using Metric = double(*)(const Point&, const Point&);

    VPTree(const std::vector<Point>& dataset, Metric metric)
      : metric(metric) {
        std::vector<Point> S = dataset;
        root = make_vp_tree(S);
      }
    ~VPTree() {
      destroy_tree(root);
    }
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

    Node* build_tree(std::vector<Point>& S) {
      if (S.empty()) {
        return nullptr;       
      }

      point p = select_vp(S);

    
    return nullptr;
    }

    Point select_vp(std::vector<Point>& S) {
      size_t sample_size = std::min<size_t>(100, size_t S.size());
      std::vector<Point>& random_sample;
      std::sample(S.begin(), S.end(), std::back_inserter(random_sample), sample_size,
                  std::mt19937 {std::random_device{}()}); // Take random sample of set S
      double best_spread = 0;
      Point best_p = S[0];

      for (const auto& p : random_sample) {
        std::vector<Point> D;
        std::sample(S.begin(), S.end(), std::back_inserter(D), sample_size,
                    std::mt19937 {std::random_device{}()});
        std::vector<double> distances;
        distances.reserve(D.size());
        for (const auto& d_point : D) {
          distances.push_back(distance_func(p, d_point));
        }

     
          // second moment E^2
          double spread = 0.0;
          for (double dist : distances) {
            double diff = dist - mu;
            spread += diff * diff;
          }
          spread /= distances.size();

          if (spread > best_spread) {
            best_spread = spread;
            best_p = p;
          }
        }
      return best_p;
    };
};

