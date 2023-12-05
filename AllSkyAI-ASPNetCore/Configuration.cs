namespace AllSkyAI_ASPNetCore
{
    public class Configuration
    {
        public string Url { get; set; }

        public string Model { get; set; }
        public string Labels { get; set; }
        public bool ConfigOk { get; set; }

        public Configuration()
        {
            Url = string.Empty;
            Model = string.Empty;
            Labels = string.Empty;
            ConfigOk = false;

            ReadConfig();
        }

        public void ReadConfig()
        {
            if (!File.Exists(@".\\config.cfg"))
            {
                Console.WriteLine("No config.cfg file.");
                Console.ReadKey();
                Environment.Exit(0);
            }

            var confFile = File.ReadAllLines(@".\\config.cfg");
            var confList = new List<string>(confFile);
            bool modelFileExists = false;
            bool labelsFileExists = false;

            foreach (var conf in confList)
            {
                if (conf.StartsWith("URL"))
                {
                    var h = conf.Split('=').Last();
                    if (string.IsNullOrEmpty(h))
                    {
                        Console.WriteLine("UEL can't be empty, check config.cfg");
                    }
                    else
                    {
                        Url = h;
                    }
                }

                else if (conf.StartsWith("MODEL"))
                {
                    var h = conf.Split('=').Last();
                    if (string.IsNullOrEmpty(h))
                    {
                        Console.WriteLine("MODEL can't be empty, check config.cfg");
                    }
                    else if(!File.Exists(".\\model\\" + h))
                    {
                        Console.WriteLine("Missing Onnx model file in ./models/ directory");
                    }
                    else
                    {
                        modelFileExists = true;
                        Model = h;
                    }
                }
                else if (conf.StartsWith("LABELS"))
                {
                    var h = conf.Split('=').Last();
                    if (string.IsNullOrEmpty(h))
                    {
                        Console.WriteLine("LABELS can't be empty, check config.cfg");
                    }
                    else if (!File.Exists(".\\model\\" + h))
                    {
                        Console.WriteLine("Missing Labels file in ./models/ directory");
                    }
                    else
                    {
                        labelsFileExists = true;
                        Labels = h;
                    }
                }
            }

            if (string.IsNullOrEmpty(Url) && string.IsNullOrEmpty(Model) && !modelFileExists)
            {
                ConfigOk = false;
            }
            else
            {
                ConfigOk = true;
            }
        }
    }
}
