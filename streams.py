from NamedStreamGenerator import NamedStreamGenerator


stream0 = NamedStreamGenerator(
    name="stream0",
    n_drifts=3,
    random_state=2222,
    n_chunks=40,
    concept_sigmoid_spacing=10,
    weights=[0.9, 0.1]
)

stream1 = NamedStreamGenerator(
    name="stream1",
    n_classes=2,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=105,
    n_chunks=100,
    chunk_size=500
)

stream2 = NamedStreamGenerator(
    name="stream2",
    n_drifts=2
)
