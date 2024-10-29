#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0];
        auto B = inputs[1];
        auto shape_a = A->getDims();
        auto rank_a = A->getRank();
        auto shape_b = B->getDims();
        auto rank_b = B->getRank();
        if(transA){
            auto temp = shape_a[rank_a-2];
            shape_a[rank_a-2] = shape_a[rank_a-1];
            shape_a[rank_a-1] = temp;
        }

        if(transB){
            auto temp = shape_b[rank_b-2];
            shape_b[rank_b-2] = shape_b[rank_b-1];
            shape_b[rank_b-1] = temp;
        }

        auto output_dims = shape_a;
        for(size_t i = 0; i<rank_a-2;i++){
            output_dims[i] = std::max(shape_a[i], shape_b[i]);
        }

        output_dims[rank_a-2] = shape_a[rank_a-2];
        output_dims[rank_a-1] = shape_b[rank_a-1];
        return {{output_dims}};
    }

} // namespace infini