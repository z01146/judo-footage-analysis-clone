import io

import numpy as np
from imageio.v3 import imread
from pyspark.ml import Transformer
from pyspark.ml.functions import array_to_vector, predict_batch_udf, vector_to_array
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from scipy.fftpack import dctn


class HasFilterSize(Params):
    filter_size = Param(
        Params._dummy(),
        "filter_size",
        "filter size to use for DCTN",
        typeConverter=TypeConverters.toListInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(filter_size=[8])

    def getFilterSize(self):
        return self.getOrDefault(self.filter_size)


class HasTensorShapes(Params):
    tensor_shape = Param(
        Params._dummy(),
        "tensor_shape",
        "shape of the tensor",
        typeConverter=TypeConverters.toListInt,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(tensor_shape=[8])

    def getTensorShape(self):
        return self.getOrDefault(self.tensor_shape)


class HasCheckpoint(Params):
    checkpoint = Param(
        Params._dummy(),
        "checkpoint",
        "path to the checkpoint",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self) -> None:
        super().__init__()
        self._setDefault(checkpoint="yolov8n.pt")

    def getCheckpoint(self):
        return self.getOrDefault(self.checkpoint)


class WrappedYOLOv8DetectEmbedding(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasTensorShapes,
    HasCheckpoint,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        checkpoint="yolov8n.pt",
        batch_size=4,
        output_tensor_shapes=None,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            tensor_shape=np.array(output_tensor_shapes).tolist(),
            checkpoint=checkpoint,
        )
        self.batch_size = batch_size

    def _make_predict_fn(self):
        """Return PredictBatchFunction using a closure over the model"""
        import threading

        from ultralytics import YOLO

        model = YOLO(self.getCheckpoint())
        lock = threading.Lock()

        def predict(inputs: np.ndarray) -> np.ndarray:
            lock.acquire()
            activations = []

            def _hook(model, input, output):
                activations.append(output.detach().cpu().numpy())

            handle = (
                model.model.model[-1]
                ._modules["cv3"]
                ._modules["2"]
                .register_forward_hook(_hook)
            )

            try:
                images = [imread(io.BytesIO(input)) for input in inputs]
                list(
                    model.predict(
                        images,
                        device="cpu",
                        stream=True,
                        save=False,
                        verbose=False,
                    )
                )
                res = np.stack(activations)
                assert list(res.shape[2:]) == list(
                    self.getTensorShape()
                ), f"Expected shape {self.getTensorShape()} but got {res.shape}"

            finally:
                handle.remove()
                lock.release()
            return res.reshape(len(images), -1)

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            array_to_vector(
                predict_batch_udf(
                    make_predict_fn=self._make_predict_fn,
                    return_type=ArrayType(FloatType()),
                    batch_size=self.batch_size,
                )(self.getInputCol())
            ),
        )


class DCTN(
    Transformer,
    HasInputCol,
    HasOutputCol,
    HasFilterSize,
    HasTensorShapes,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    """
    Run n-dimensional DCT on the input column
    """

    def __init__(
        self,
        input_col: str = "input",
        output_col: str = "output",
        filter_size: list = None,
        input_tensor_shapes: list = None,
        batch_size: int = 8,
    ):
        super().__init__()
        self._setDefault(
            inputCol=input_col,
            outputCol=output_col,
            filter_size=np.array(filter_size).tolist(),
            tensor_shape=np.array(input_tensor_shapes).tolist(),
        )
        self.batch_size = batch_size
        print(f"Filter size: {self.getFilterSize()}")
        print(f"Input tensor shapes: {self.getTensorShape()}")

    def _make_predict_fn(self):
        def dctn_filter(tile):
            coeff = dctn(tile)
            # slice the coefficients using the filter size
            for axis, k in enumerate(self.getFilterSize()):
                coeff = np.take(coeff, np.arange(k), axis=axis)
            flat = coeff.flatten()
            # make sure the output is the right size
            res = np.zeros(np.prod(self.getFilterSize()))
            res[: len(flat)] = flat
            return res

        def predict(inputs: np.ndarray) -> np.ndarray:
            return np.array([dctn_filter(x) for x in inputs])

        return predict

    def _transform(self, df: DataFrame):
        return df.withColumn(
            self.getOutputCol(),
            array_to_vector(
                predict_batch_udf(
                    make_predict_fn=self._make_predict_fn,
                    return_type=ArrayType(FloatType()),
                    batch_size=self.batch_size,
                    input_tensor_shapes=[self.getTensorShape()],
                )(vector_to_array(self.getInputCol())),
            ),
        )
