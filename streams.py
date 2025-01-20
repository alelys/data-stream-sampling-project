from NamedStreamGenerator import NamedStreamGenerator


stream1 = NamedStreamGenerator(
    name="stream1",
    n_drifts=3,
    random_state=2222,
    n_chunks=40,
    concept_sigmoid_spacing=10,
    weights=[0.9, 0.1]
)


stream2 = NamedStreamGenerator(
    name="stream2",
    weights=[0.3, 0.7],
    n_chunks=40,
    random_state=2222
)

stream3 = NamedStreamGenerator(
    name="stream3", 
    weights=(2, 5, 0.9),
    n_drifts=3,
    concept_sigmoid_spacing=5,
    recurring=True,
    incremental=True,
    n_chunks=40,
    random_state=2222
)
