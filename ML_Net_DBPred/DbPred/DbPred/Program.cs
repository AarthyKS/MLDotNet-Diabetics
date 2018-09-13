using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace DbPred
{
    class Program
    {
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "data", "train1.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "data", "test1.csv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "data", "Model.zip");
        static async Task Main(string[] args)
        {
            var model = await Train();
            Evaluate(model);
            Predict(model);

            Console.ReadKey();
        }

        public static async Task<PredictionModel<DiabetesData, DiabetesPrediction>> Train()
        {
            var pipeline = new LearningPipeline();
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<DiabetesData>(useHeader: false, separator: ','));
            //pipeline.Add(new MissingValueSubstitutor("num_preg") { ReplacementKind = NAReplaceTransformReplacementKind.Mean });
            pipeline.Add(new ColumnConcatenator("Features", "num_preg", "glucose_conc", "diastolic_bp",
                                                    "thickness", "insulin", "bmi", "diab_pred", "age"));
            pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 10, NumTrees = 10, MinDocumentsInLeafs = 2 });
            PredictionModel<DiabetesData, DiabetesPrediction> model = pipeline.Train<DiabetesData, DiabetesPrediction>();
            await model.WriteAsync(_modelpath);
            return model;

        }

        public static void Evaluate(PredictionModel<DiabetesData, DiabetesPrediction> model)
        {
            var testData = new TextLoader(_testDataPath).CreateFrom<DiabetesData>(separator: ',');
            var evaluator = new BinaryClassificationEvaluator();
            BinaryClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }

        public static void Predict(PredictionModel<DiabetesData, DiabetesPrediction> model)
        {
            var lines = File.ReadAllLines(_testDataPath);
            List<DiabetesData> lstData = new List<DiabetesData>();
            foreach (var line in lines)

            {

                string[] words = line.Split(',');

                foreach (var word in words)

                {
                    List<DiabetesData> data = new List<DiabetesData>();
                    DiabetesData diabData = new DiabetesData();
                    //num_preg,glucose_conc,diastolic_bp,thickness,insulin,bmi,diab_pred,age,skin,diabetes
                    // Stuck here how to proceed as I tried to assign it I get paragraphs of errors and exceptions
                    diabData.num_preg = Convert.ToSingle(words[0]);
                    diabData.glucose_conc = Convert.ToSingle(words[1]);
                    diabData.diastolic_bp = Convert.ToSingle(words[2]);
                    diabData.thickness = Convert.ToSingle(words[3]);
                    diabData.insulin = Convert.ToSingle(words[4]);
                    diabData.bmi = Convert.ToSingle(words[5]);
                    diabData.diab_pred = Convert.ToSingle(words[6]);
                    diabData.age = Convert.ToSingle(words[7]);
                    diabData.diabetes = Convert.ToInt32(words[8]) == 1;

                    if(new Random().Next() % 50 == 0)
                    lstData.Add(diabData);
                }
            }
            IEnumerable<DiabetesPrediction> predictions = model.Predict(lstData);

            Console.WriteLine();
            Console.WriteLine(" Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = lstData.Zip(predictions, (diabetes, prediction) => (diabetes, prediction));

            int total = sentimentsAndPredictions.Count();
            int rightCount = 0, wrongCount = 0;
            foreach (var item in sentimentsAndPredictions)
            {
                if (item.diabetes.diabetes == item.prediction.diabetes)
                    Console.WriteLine($"Data: {item.diabetes.diabetes} ------ Prediction: {item.prediction.diabetes}");
            }
            Console.WriteLine();
        }
    }
}
