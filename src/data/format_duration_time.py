

def format_duration(total_seconds):

    if total_seconds <= 0:
        return "less 1 second"

    days = int(total_seconds // 86400)
    remaining_seconds = total_seconds % 86400
    hours = int(remaining_seconds // 3600)
    remaining_seconds %= 3600
    minutes = int(remaining_seconds // 60)
    seconds = remaining_seconds % 60

    # สร้าง List เพื่อเก็บข้อความแต่ละส่วน
    parts = []
    if days > 0:
        parts.append(f"{days} day")
    if hours > 0:
        parts.append(f"{hours} hours")
    if minutes > 0:
        parts.append(f"{minutes} minutes")
    
    parts.append(f"{seconds:.2f} second")
    return ", ".join(parts)

