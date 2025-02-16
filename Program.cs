using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SentimentAnalysisML
{
    public class SentimentData
    {
        [LoadColumn(0)] // Text column
        public string Text { get; set; } = String.Empty;

        [LoadColumn(1)] // Sentiment label
        public string Sentiment { get; set; } = String.Empty;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")] // Prediction output
        public string Prediction { get; set; } = String.Empty;
    }

    class Program
    {
        static void Main(string[] args)
        {
            // Step 1: Create ML Context
            MLContext mlContext = new MLContext();

            // Step 2: Load Data
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>("sentiment_data.csv", separatorChar: ',', hasHeader: true);

            Console.WriteLine("Data Loaded Successfully!");

            var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(SentimentData.Sentiment)))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var trainingPipeline = dataPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(dataView);

            Console.WriteLine("Model Trained Successfully!");

            // Evaluate the model
            var predictions = trainedModel.Transform(dataView);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score", "PredictedLabel");

            Console.WriteLine($"Accuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");

        }
    }


}