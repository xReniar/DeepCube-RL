import { launch } from "puppeteer";
import { writeFileSync } from "fs"

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getSolution(page, scramble) {
    const moveMappings = {
        // Clockwise
        'U': 'uu', 'L': 'll', 'R': 'rr',
        'D': 'dd', 'F': 'ff', 'B': 'bb',
        // Counter Clockwise
        "U'": 'ui', "L'": 'li', "R'": 'ri',
        "D'": 'di', "F'": 'fi', "B'": 'bi',
    };

    // execute scramble to solved cube
    const moves = scramble.trim().split(/\s+/);
    for (const move of moves) {
        await page.evaluate((fnName) => {
            if (typeof window[fnName] === 'function') {
                window[fnName]();
                kiir();
            }
        }, moveMappings[move]);
        sleep(100);
    }

    // generate solution
    await page.evaluate(() => {
        thinkAndSolve();
    });

    // after solution is generated start reading it
    const display = await page.waitForSelector('#solutiondisplay', { visible: true });
    await display.waitForSelector('span');
    
    const child = await display.$$("span");
    const solution = [];
    for (const span of child) {
        solution.push(await span.evaluate(el => el.textContent.trim()));
    }
    solution.shift()

    return solution;
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
const browser = await launch();

(async () => {
    const data = {};
    let counter = 1;

    // new version
    const concurrency = 16;
    const datasetArray = Array.from(dataset);
    for(let i = 0;i < datasetArray.length; i += concurrency){
        const batch = datasetArray.slice(i, i + concurrency);

        const results = await Promise.all(batch.map(async (scramble) => {
            const newPage = await browser.newPage();
            await newPage.goto("https://solverubikscube.com/", { waitUntil: 'domcontentloaded' })

            const solution = await getSolution(newPage, scramble);
            await newPage.close();
            return { scramble, solution: solution.join(" ") };
        }));

        for (const result of results) {
            data[counter] = result;
            counter++;
        }
        sleep(250);
    }

    writeFileSync('data.json', JSON.stringify(data, null, 2));
    await browser.close();
})();