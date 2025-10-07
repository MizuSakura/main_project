using Avalonia.Controls;
using Avalonia.Interactivity;
using System;
using System.Net.Http;
using System.Threading.Tasks;

// **สำคัญ:** เช็คให้แน่ใจว่า Namespace ตรงกับโปรเจกต์ของนาย!
namespace MyAvaloniaApp; 

public partial class MainWindow: Window
{
    private static readonly HttpClient client = new HttpClient();

    public MainWindow()
    {
        InitializeComponent();
        client.BaseAddress = new Uri("http://127.0.0.1:8000"); 
        AppendLog("System Ready. Waiting for command.");
    }
    
    // ฟังก์ชันกลางสำหรับส่งคำสั่งและรอรับผล
    private async Task ExecuteBackendProcess(string endpoint)
    {
        AppendLog($"Sending command to endpoint: {endpoint}...");
        try
        {
            var response = await client.PostAsync(endpoint, null);

            // อ่านเนื้อหาตอบกลับ ไม่ว่าจะสำเร็จหรือล้มเหลว
            var responseBody = await response.Content.ReadAsStringAsync();

            if (response.IsSuccessStatusCode)
            {
                // ถ้าสำเร็จ (โค้ด 2xx)
                AppendLog($"✅ SUCCESS from Backend: {responseBody}");
            }
            else
            {
                // ถ้าล้มเหลว ให้แสดง Error ที่ FastAPI ส่งมาโดยตรง!
                AppendLog($"❌ FAILED with status {response.StatusCode}. Backend says:\n{responseBody}");
            }
        }
        catch (Exception ex)
        {
            // ถ้าเกิด Error ร้ายแรง (เช่น Python ดับ)
            AppendLog($"❌ CRITICAL ERROR: Could not send command. Is the Python server running? ({ex.Message})");
        }
    }

    private async void StartScalingButton_Click(object sender, RoutedEventArgs e)
    {
        await ExecuteBackendProcess("/start-scaling");
    }

    private async void StartTrainingButton_Click(object sender, RoutedEventArgs e)
    {
        await ExecuteBackendProcess("/start-training");
    }

    private async void StartSimulationButton_Click(object sender, RoutedEventArgs e)
    {
        await ExecuteBackendProcess("/start-simulation");
    }

    // ฟังก์ชันสำหรับแสดง Log บนหน้าจอ
    private void AppendLog(string message)
    {
        var logTextBlock = this.FindControl<TextBlock>("LogTextBlock");
        if (logTextBlock != null)
        {
            logTextBlock.Text += $"[{DateTime.Now:HH:mm:ss}] {message}\n\n";
        }
    }
}