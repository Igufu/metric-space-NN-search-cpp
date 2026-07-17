#include <iostream>
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <limits>

template <size_t V>
class VPTree {
public:
    using Point = std::array<double, V>;
    using Metric = double(*)(const Point&, const Point&);

    VPTree(const std::vector<Point>& dataset, Metric metric)
        : metric(metric) {
        std::vector<Point> S = dataset;
        root = build_tree(S);
    }

    ~VPTree() {
        destroy_tree(root);
    }

    bool nearest(const Point& q, Point& out, double& out_dist) const {
        if (!root) {
            return false;
        }
        double tau = std::numeric_limits<double>::infinity();
        const Node* best = nullptr;
        search(root, q, tau, best);
        if (!best) {
            return false;
        }
        out = best->p;
        out_dist = tau;
        return true;
    }

private:
    struct Node {
        Point p;
        double mu;
        Node *left, *right;

        Node(const Point& vantage_point, double median)
            : p(vantage_point), mu(median), left(nullptr), right(nullptr) {}
    };

    Node* root = nullptr;
    Metric metric;

    void destroy_tree(Node* node) {
        if (!node) {
            return;
        }
        destroy_tree(node->left);
        destroy_tree(node->right);
        delete node;
    }

    Node* build_tree(std::vector<Point>& S) {
        if (S.empty()) {
            return nullptr;
        }

        Point p = select_vp(S);

        auto it = std::find(S.begin(), S.end(), p);
        if (it != S.end()) {
            S.erase(it);
        }

        if (S.empty()) {
            return new Node(p, 0.0);
        }

        std::vector<double> distances;
        distances.reserve(S.size());
        for (const auto& s : S) {
            distances.push_back(metric(p, s));
        }

        size_t mid = distances.size() / 2;
        std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
        double mu = distances[mid];

        std::vector<Point> L, R;
        for (const auto& s : S) {
            if (metric(p, s) < mu) {
                L.push_back(s);
            } else {
                R.push_back(s);
            }
        }

        Node* node = new Node(p, mu);
        node->left = build_tree(L);
        node->right = build_tree(R);

        return node;
    }

    Point select_vp(std::vector<Point>& S) {
        size_t sample_size = std::min<size_t>(100, S.size());
        std::vector<Point> random_sample;
        std::sample(S.begin(), S.end(), std::back_inserter(random_sample), sample_size,
                    std::mt19937 {std::random_device{}()});
        double best_spread = 0;
        Point best_p = S[0];

        for (const auto& p : random_sample) {
            std::vector<Point> D;
            std::sample(S.begin(), S.end(), std::back_inserter(D), sample_size,
                        std::mt19937 {std::random_device{}()});
            std::vector<double> distances;
            distances.reserve(D.size());
            for (const auto& d_point : D) {
                distances.push_back(metric(p, d_point));
            }
            if (distances.empty()) {
                continue;
            }

            size_t mid = distances.size() / 2;
            std::nth_element(distances.begin(), distances.begin() + mid, distances.end());
            double mu = distances[mid];

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
    }

    void search(const Node* node, const Point& q, double& tau, const Node*& best) const {
        if (!node) {
            return;
        }

        double dist = metric(node->p, q);
        if (dist < tau) {
            tau = dist;
            best = node;
        }

        if (dist < node->mu) {
            search(node->left, q, tau, best);
            if (dist + tau >= node->mu) {
                search(node->right, q, tau, best);
            }
        } else {
            search(node->right, q, tau, best);
            if (dist - tau < node->mu) {
                search(node->left, q, tau, best);
            }
        }
    }
};

template <size_t V>
double euclidean(const std::array<double, V>& a, const std::array<double, V>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < V; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

