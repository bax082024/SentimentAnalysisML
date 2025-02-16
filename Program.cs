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
        public string Prediction { get; set; }
    }



}