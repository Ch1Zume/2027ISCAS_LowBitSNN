from test_neuron import (
    test_constant_input_response,
    test_threshold_effect,
    test_tau_effect,
    test_reset_mode_difference,
    test_batch_state_independence,
)

from test_layer import (
    test_conv2d_snn_shape_and_equivalence,
    test_linear_snn_shape_and_equivalence,
    test_pooling_shape,
    test_batchnorm2d_shape,
    test_batchnorm1d_shape,
    test_flatten_snn,
)

from test_encoder import (
    test_repeat_encoder,
    test_poisson_encoder,
    test_latency_encoder,
)


def main():
    print("========== Running SNN Testbench ==========")

    # neuron
    test_constant_input_response(visualize=True)
    test_threshold_effect(visualize=True)
    test_tau_effect(visualize=True)
    test_reset_mode_difference(visualize=True)
    test_batch_state_independence(visualize=True)

    # layers
    test_conv2d_snn_shape_and_equivalence()
    test_linear_snn_shape_and_equivalence()
    test_pooling_shape()
    test_batchnorm2d_shape()
    test_batchnorm1d_shape()
    test_flatten_snn()

    # encoders
    test_repeat_encoder()
    test_poisson_encoder()
    test_latency_encoder()

    print("\n========== All tests passed ==========")


if __name__ == "__main__":
    main()