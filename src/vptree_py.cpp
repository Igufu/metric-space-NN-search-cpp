#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "vptree.cpp"

namespace py = pybind11;

constexpr size_t DIM = 784;

using MnistPoint = std::array<double, DIM>;

class MnistVPTree {
public:
    explicit MnistVPTree(py::array_t<double, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != DIM) {
            throw std::runtime_error("Expected shape (N, 784)");
        }
        const size_t N = static_cast<size_t>(buf.shape[0]);
        const double* ptr = static_cast<const double*>(buf.ptr);

        std::vector<MnistPoint> points;
        points.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            MnistPoint p;
            std::memcpy(p.data(), ptr + i * DIM, DIM * sizeof(double));
            points.push_back(p);
        }
        n_ = N;
        tree_ = std::make_unique<VPTree<DIM>>(points, &euclidean<DIM>);
    }

    py::tuple nearest(py::array_t<double, py::array::c_style | py::array::forcecast> query) const {
        auto buf = query.request();
        if (static_cast<size_t>(buf.size) != DIM) {
            throw std::runtime_error("Expected shape (784,)");
        }
        MnistPoint q;
        std::memcpy(q.data(), buf.ptr, DIM * sizeof(double));

        MnistPoint out;
        double dist = 0.0;
        bool ok = tree_->nearest(q, out, dist);
        if (!ok) {
            return py::make_tuple(py::none(), py::none());
        }
        py::array_t<double> out_arr(static_cast<py::ssize_t>(DIM));
        std::memcpy(out_arr.request().ptr, out.data(), DIM * sizeof(double));
        return py::make_tuple(out_arr, dist);
    }

    py::array_t<double> nearest_batch(py::array_t<double, py::array::c_style | py::array::forcecast> queries) const {
        auto buf = queries.request();
        if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != DIM) {
            throw std::runtime_error("Expected shape (M, 784)");
        }
        const size_t M = static_cast<size_t>(buf.shape[0]);
        const double* ptr = static_cast<const double*>(buf.ptr);

        py::array_t<double> distances(static_cast<py::ssize_t>(M));
        double* d_ptr = static_cast<double*>(distances.request().ptr);

        MnistPoint q;
        MnistPoint out;
        for (size_t i = 0; i < M; ++i) {
            std::memcpy(q.data(), ptr + i * DIM, DIM * sizeof(double));
            double dist = 0.0;
            bool ok = tree_->nearest(q, out, dist);
            d_ptr[i] = ok ? dist : std::numeric_limits<double>::quiet_NaN();
        }
        return distances;
    }

    size_t size() const { return n_; }
    size_t dim() const { return DIM; }

private:
    std::unique_ptr<VPTree<DIM>> tree_;
    size_t n_ = 0;
};

PYBIND11_MODULE(vptree_mnist, m) {
    m.doc() = "pybind11 bindings for VPTree<784> with Euclidean metric";
    py::class_<MnistVPTree>(m, "VPTree")
        .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>>(),
             py::arg("data"),
             "Build a VP-tree from an (N, 784) float64 array.")
        .def("nearest", &MnistVPTree::nearest, py::arg("query"),
             "Return (nearest_point, distance) for a (784,) query.")
        .def("nearest_batch", &MnistVPTree::nearest_batch, py::arg("queries"),
             "Return an array of distances to the nearest neighbor for each (M, 784) query row.")
        .def_property_readonly("size", &MnistVPTree::size)
        .def_property_readonly("dim", &MnistVPTree::dim);
}
