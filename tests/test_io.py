from typesymbolic._io import atomic_write


def test_atomic_write_creates_the_file_with_the_given_content(tmp_path):
    path = tmp_path / "sub" / "file.txt"
    atomic_write(path, b"hello")
    assert path.read_bytes() == b"hello"


def test_atomic_write_leaves_no_tmp_file_behind(tmp_path):
    path = tmp_path / "file.txt"
    atomic_write(path, b"hello")
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "file.txt"
    atomic_write(path, b"first")
    atomic_write(path, b"second")
    assert path.read_bytes() == b"second"


def test_repeated_writes_to_the_same_path_each_leave_exactly_one_file(tmp_path):
    """`tempfile.mkstemp` names each call's tmp file uniquely, so two writes
    to the same path in one process never collide."""
    path = tmp_path / "file.txt"
    for i in range(5):
        atomic_write(path, str(i).encode())
    assert path.read_bytes() == b"4"
    assert list(tmp_path.iterdir()) == [path]
