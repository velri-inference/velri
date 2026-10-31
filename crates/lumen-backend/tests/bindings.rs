use lumen_backend::{BindingKind, BindingSpec, ExecutionBindings, RowExtent};
use lumen_core::{DType, Shape};
use lumen_mem::BufferId;
use lumen_tensor::TensorRef;

#[test]
fn malformed_public_bindings_are_rejected_before_a_step() {
    for (kind, extent, dtype, width, rows) in [
        (BindingKind::Tokens, RowExtent::Fixed(1), DType::U32, 1, 1),
        (BindingKind::Tokens, RowExtent::Step, DType::U32, 2, 4),
        (BindingKind::Sequence, RowExtent::Step, DType::F32, 2, 4),
        (
            BindingKind::Activation,
            RowExtent::History,
            DType::U32,
            2,
            8,
        ),
    ] {
        let spec = BindingSpec {
            tensor: TensorRef {
                id: BufferId(0),
                dtype,
                shape: Shape::from([rows, width]),
            },
            row_width: width,
            extent,
            kind,
            initial: None,
        };
        assert!(
            ExecutionBindings::new(&[spec], 4, 8).is_err(),
            "{kind:?} / {extent:?}"
        );
    }
}

#[test]
fn matching_element_count_does_not_allow_conflicting_tensor_axes() {
    for (kind, extent, dtype, shape, width, max_tokens, max_context) in [
        (
            BindingKind::Activation,
            RowExtent::Step,
            DType::F32,
            Shape::from([2, 2]),
            4,
            1,
            8,
        ),
        (
            BindingKind::Sequence,
            RowExtent::History,
            DType::F32,
            Shape::from([2, 4]),
            2,
            1,
            4,
        ),
        (
            BindingKind::Tokens,
            RowExtent::Step,
            DType::U32,
            Shape::from([2, 1]),
            1,
            2,
            8,
        ),
    ] {
        let spec = BindingSpec {
            tensor: TensorRef {
                id: BufferId(0),
                dtype,
                shape,
            },
            row_width: width,
            extent,
            kind,
            initial: None,
        };
        assert!(
            ExecutionBindings::new(&[spec], max_tokens, max_context).is_err(),
            "{kind:?}"
        );
    }
}
