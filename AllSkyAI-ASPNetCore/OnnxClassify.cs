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
using System;

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

        string[] Labels;

        public OnnxClassify()
        {
            _logger = LoggerFactory.Create(options => { }).CreateLogger<OnnxClassify>();
            string modelFilePath = ".\\model\\" + configuration.Model;
            if (!File.Exists(modelFilePath))
            {
                Console.WriteLine($"Model file does not exists: {modelFilePath}");
                return;
            }

            string labelsFilePath = ".\\model\\" + configuration.Labels;
            Labels = LabelMap.readLabels(labelsFilePath);
            
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
            int channels = dimentions[3];

            // Read image
            Image<Rgb24> image = Image.Load<Rgb24>(saveLocation);
            
            if(channels == 1)
            {
                image.Mutate(x => x.Grayscale());
            }
            
            if(height == width) // We need to reize to height and crop
            {
                image.Mutate(x =>
                    x.Resize(
                        new ResizeOptions()
                        {
                            Mode = ResizeMode.Max,
                            Size = new Size(int.MaxValue, height)
                        }
                    )
                );

                var size = image.Size();
                var l = (size.Width / 2)-256;
                var t = 0;
                var r = (size.Width / 2) + 256;
                var b = size.Height;

                image.Mutate(x => x.Crop(Rectangle.FromLTRB(l, t, r, b)));
            }
            // Resize to fix input dimentions
            else
            {
                image.Mutate(x => x.Resize(width, height, KnownResamplers.Lanczos3));
            }

            // Preprocess image
            Tensor<float> input = new DenseTensor<float>(new[] { 1, height, width, channels });
            if(channels == 1)
            {
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
            }
            else
            {
                image.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < accessor.Height; y++)
                    {
                        Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                        for (int x = 0; x < accessor.Width; x++)
                        {
                            input[0, y, x, 0] = ((pixelSpan[x].R));
                            input[0, y, x, 1] = ((pixelSpan[x].G));
                            input[0, y, x, 2] = ((pixelSpan[x].B));
                        }
                    }
                });
            }
            

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
            IEnumerable<Prediction> classificationScore = softmax.Select((x, i) => new Prediction { Label = Labels[i], Confidence = x })
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

