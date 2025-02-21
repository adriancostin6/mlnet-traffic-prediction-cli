namespace ciucli;
using Microsoft.ML.Data;
using Microsoft.ML;

class TrafficData 
{
    [LoadColumn(0)] 
    public string? DateTime { get; set; }
    [LoadColumn(1)] 
    public float Junction { get; set; }
    [LoadColumn(2)] 
    public float Vehicles { get; set; }
    [LoadColumn(3)] 
    public float Id { get; set; }
}

class Prediction 
{
    [ColumnName("Score")]
    public float PredictedTrafficVolume;
}

// TODO extra: Make these configurable via file selector as well
class Data {
    public static string csvPath =
        "/home/adrianc/repos/aspnet/ciucli/traffic.csv";
    public static string modelPath =
        "/home/adrianc/repos/aspnet/ciucli/traffic_model.zip";

    public static MLContext mlCtx = new MLContext();
}

class Trainer {
    public static void TrainModel() 
    {
        IDataView data = Data.mlCtx.Data.LoadFromTextFile<TrafficData>(
                Data.csvPath,
                separatorChar: ',', 
                hasHeader: true
        );
        Console.WriteLine($"Read traffic data from {Data.csvPath}");

        var pipeline = Data.mlCtx.Transforms.CopyColumns("Label", "Vehicles")
            .Append(Data
                .mlCtx.Transforms
                .Conversion.ConvertType(
                    "Junction", 
                    outputKind: DataKind.Single
                )
            )
            .Append(Data
                .mlCtx.Transforms
                .Text.FeaturizeText("DateTimeFeaturized", "DateTime"
                )
            )
            .Append(Data
                .mlCtx.Transforms
                .Concatenate(
                    "Features", 
                    new[] { "Junction", "DateTimeFeaturized"}
                )
            )
            .Append(Data
                .mlCtx.Regression
                .Trainers.FastTree(
                    labelColumnName: "Label", 
                    featureColumnName: "Features"
                )
            );

        Console.WriteLine("Training model.");
        var model = pipeline.Fit(data);
        Data
            .mlCtx.Model
            .Save(model, data.Schema, Data.modelPath);
        Console.WriteLine($"Done. Saved trained model to {Data.modelPath}");
    }
}

class Predictor {
    public static void Predict(string userInput) 
    {
        var ctx = new MLContext();

        ITransformer model = Data
            .mlCtx.Model
            .Load(Data.modelPath, out var modelSchema);
        var predictionEngine =  Data
            .mlCtx.Model
            .CreatePredictionEngine<TrafficData, Prediction>(model);

        var traffic = new TrafficData
        {
            DateTime = userInput,
            Junction = 1
        };

        var prediction = predictionEngine.Predict(traffic);
        Console.WriteLine($@"Traffic data prediction
            for junction {traffic.Junction} 
            at time {traffic.DateTime} 
            is {prediction.PredictedTrafficVolume} vehicles."
        );
    }
}

class Cli {
    public static string getUserInput() {
        Console.WriteLine("Enter date (yy-mm-dd):");
        string? date = Console.ReadLine();
        Console.WriteLine("Enter time (hh:mm:ss):");
        string? time = Console.ReadLine();

        if (date == null || time == null) {
            throw new ArgumentException("Enter date and time.");
        }

        return $"{date} {time}";
    }
}

class Program
{
    static void Main(string[] args)
    {
        if (!File.Exists(Data.modelPath)) {
            Trainer.TrainModel();
        }
        string input = Cli.getUserInput();
        Predictor.Predict(input);
    }
}
