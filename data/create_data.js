import { launch } from "puppeteer";

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function executeRubikMove(page, scramble) {
    const moveMappings = {
        // Clockwise
        'U': 'uu', 'L': 'll', 'R': 'rr',
        'D': 'dd', 'F': 'ff', 'B': 'bb',
        // Counter Clockwise
        "U'": 'ui', "L'": 'li', "R'": 'ri',
        "D'": 'di', "F'": 'fi', "B'": 'bi',
    };

    const moves = scramble.trim().split(/\s+/);
    for (const move of moves) {
        const functionName = moveMappings[move];
        await page.evaluate((fnName) => {
            if (typeof window[fnName] === 'function') {
                window[fnName]();
                kiir();
            }
        }, functionName);
        sleep(100);
    }
}

const moves = ['R', "R'", 'L', "L'", 'U', "U'", 'D', "D'", 'F', "F'", 'B', "B'"];
const dataset = new Set();

// create random scrambles
for (let i = 0; i < 10000; i++) {
    const randomSequence = Array.from({ length: 20 }, () => {
        return moves[Math.floor(Math.random() * moves.length)];
    }).join(' ');
    
    dataset.add(randomSequence);
}

// start browsing
const browser = await launch({
    headless: false,
    defaultViewport: null,
    args: ['--start-maximized']
});

const page = await browser.newPage();
await page.goto("https://solverubikscube.com/", { waitUntil: 'domcontentloaded' })
    .then(async () => {
        //console.log(scramble);
        for (const scramble of dataset) {
            console.log(scramble);
            await executeRubikMove(page, scramble);
            await sleep(1000);
        }
    }).catch((reason) => {
        console.log(reason);
    });