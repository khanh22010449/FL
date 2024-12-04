import jax
import jax.numpy as jnp

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, LinearPartitioner
from PIL import Image

key = jax.random.PRNGKey(0)

fds = None


def load_data(partition_id: int, num_partitions: int):
    global fds
    partitioner = LinearPartitioner(num_partitions=num_partitions)

    fds = FederatedDataset(
        dataset="uoft-cs/cifar10", partitioners={"train": partitioner}
    )

    partition = fds.load_partition(partition_id=partition_id)

    print(partition["img"][:5])

    partition_train_test = partition.train_test_split(
        test_size=0.25, seed=42, shuffle=True
    )

    def apply_numpy(batch):
        print("apply_numpy<<<<<----------")

        batch["img"] = [jnp.array(img) / 255.0 for img in batch["img"]]
        print(batch["img"])
        return batch

    partition_train_test.with_transform(apply_numpy)


if __name__ == "__main__":
    load_data(3, 10)
