import { launch } from "puppeteer";


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
page.goto("https://solverubikscube.com/", { waitUntil: 'domcontentloaded' }).then((value) => {
    
}).catch((reason) => {
    console.log(reason);
});