def boxCoordinates(i, xLim, yLim, rows, cols):
    offsetX = xLim / cols
    offsetY = yLim / rows

    goalX = i % cols * offsetX
    goalY = (i // 3) * offsetY

    return goalX, goalY

def main():
    for i in range(12):
        x, y, = boxCoordinates(i, 700, 600, 3, 4)
        print(f"x: {x}  y: {y}")

if __name__ == "__main__":
    main()