using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System.Net;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using AllSkyAI_ASPNetCore.Controllers;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Globalization;

namespace AllSkyAI_ASPNetCore
{
    public class Reply
    {
        public string classification { get; set; }
        public float confidence { get; set; }

        public double inference { get; set; }
    }

    public class OnnxClassify
    {
        private readonly ILogger _logger;
        System.Diagnostics.Stopwatch watch = new System.Diagnostics.Stopwatch();
        InferenceSession session;
        Configuration configuration = new Configuration();

        CultureInfo cultureInfo = CultureInfo.GetCultureInfo("en-US");

        public OnnxClassify()
        {
            _logger = LoggerFactory.Create(options => { }).CreateLogger<OnnxClassify>();
            string modelFilePath = ".\\model\\" + configuration.Model;
            session = new InferenceSession(modelFilePath);
        }

        public string ClassifyImage(string url)
        {
            string saveLocation = @".\\tmp.jpg";

            Console.WriteLine($"Downloading AllSkyImage: {url}");
            _logger.Log(Microsoft.Extensions.Logging.LogLevel.Information, $"Downloading AllSkyImage: {url}");

            watch.Reset();
            watch.Start();

            HttpWebRequest lxRequest = (HttpWebRequest)WebRequest.Create(url);
            String lsResponse = string.Empty;
            using (HttpWebResponse lxResponse = (HttpWebResponse)lxRequest.GetResponse())
            {
                using (BinaryReader reader = new BinaryReader(lxResponse.GetResponseStream()))
                {
                    Byte[] lnByte = reader.ReadBytes(1 * 1024 * 1024 * 10);
                    using (FileStream lxFS = new FileStream(saveLocation, FileMode.Create))
                    {
                        lxFS.Write(lnByte, 0, lnByte.Length);
                    }
                }
            }
            watch.Stop();
            Console.WriteLine($"Download done in {TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds).TotalSeconds}s, classifying image...");

            watch.Reset();
            watch.Start();

            // Load Metadata
            var metaData = session.InputMetadata.First().Value;
            var dimentions = metaData.Dimensions.ToList();
            int height = dimentions[1];
            int width = dimentions[2];

            // Read image
            Image<Rgb24> image = Image.Load<Rgb24>(saveLocation);
            image.Mutate(x => x.Resize(width, height, KnownResamplers.Lanczos3));
            image.Mutate(x => x.Grayscale());

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, height, width, 1 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input[0, y, x, 0] = ((pixelSpan[x].R));
                    }
                }
            });

            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", input)
            };

            // Run inference
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract classes
            IEnumerable<Prediction> classificationScore = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                               .OrderByDescending(x => x.Confidence)
                               .Take(1);
            watch.Stop();

            float confidence = classificationScore.First().Confidence * 100f;

            Console.WriteLine($"Label: {classificationScore.First().Label}, Confidence: {classificationScore.First().Confidence}");
            Console.WriteLine($"Classification done in {TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds).TotalSeconds}s");

            var result = new Reply { 
                classification = classificationScore.First().Label,
                confidence = confidence,
                inference = TimeSpan.FromMilliseconds(watch.ElapsedMilliseconds).TotalSeconds
            };
            return JsonSerializer.Serialize(result);
        }

    }
}

