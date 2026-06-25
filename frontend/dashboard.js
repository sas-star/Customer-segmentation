document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------
    // 1. Elements
    // ------------------------------------------------
    const kpiQProfit = document.getElementById("kpi-q-profit");
    const kpiProfitLift = document.getElementById("kpi-profit-lift");
    const kpiWasteSaved = document.getElementById("kpi-waste-saved");
    const kpiWasteReduction = document.getElementById("kpi-waste-reduction");
    
    const simCategory = document.getElementById("sim-category");
    const simSupply = document.getElementById("sim-supply");
    const simDemand = document.getElementById("sim-demand");
    
    const link1 = document.getElementById("link-1");
    const link2 = document.getElementById("link-2");
    const link3 = document.getElementById("link-3");
    
    const heatmapGrid = document.getElementById("q-table-heatmap");
    const heatmapEx = document.getElementById("heatmap-explanation");
    
    // Globals to store loaded data
    let globalQTable = null;
    let globalPolicy = null;
    
    const actionsDesc = [
        "Maintain Standard",
        "Discount Price (-15%)",
        "Refrigerate Surplus",
        "Repurpose Surplus"
    ];
    
    const actionsExplain = [
        "Standard price matching. Best when supply matches demand perfectly.",
        "Discounting prices by 15% to stimulate consumer demand and clear high stocks.",
        "Preserving stock through enhanced refrigeration to reduce spoilage. Incurs energy holding cost.",
        "Repurposing surplus stock into alternative recipes. Generates value while preventing direct waste."
    ];

    // Helper formatting functions
    const formatCurrency = (val) => {
        return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
    };

    // ------------------------------------------------
    // 2. Fetch Prediction Batch Data & Render Charts
    // ------------------------------------------------
    fetch("/api/predict")
        .then(res => res.json())
        .then(data => {
            // Update KPIs
            kpiQProfit.innerText = formatCurrency(data.summary.q_learning_profit);
            
            const lift = ((data.summary.q_learning_profit - data.summary.baseline_profit) / data.summary.baseline_profit) * 100;
            kpiProfitLift.innerText = `+${lift.toFixed(1)}% lift vs. Baseline ($${(data.summary.baseline_profit/1000).toFixed(0)}k)`;
            
            const wasteSaved = data.summary.baseline_waste - data.summary.q_learning_waste;
            kpiWasteSaved.innerText = `${Math.round(wasteSaved)} units`;
            kpiWasteReduction.innerText = `-${data.summary.waste_reduction_pct}% waste reduction`;

            // CHART 1: Profit Growth Comparison (Line Chart)
            const ctxProfit = document.getElementById("profitComparisonChart").getContext("2d");
            new Chart(ctxProfit, {
                type: 'line',
                data: {
                    labels: data.profits_comparison.index,
                    datasets: [
                        {
                            label: 'Q-Learning Optimized',
                            data: data.profits_comparison.q_learning,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.04)',
                            borderWidth: 2.5,
                            pointRadius: 0,
                            fill: true,
                            tension: 0.1
                        },
                        {
                            label: 'Standard Baseline',
                            data: data.profits_comparison.baseline,
                            borderColor: '#ef4444',
                            backgroundColor: 'rgba(239, 68, 68, 0.04)',
                            borderWidth: 1.8,
                            pointRadius: 0,
                            fill: true,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 11 } } }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } },
                        y: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });

            // CHART 2: Supply vs Demand Alignment (Line Chart)
            const ctxSD = document.getElementById("supplyDemandChart").getContext("2d");
            new Chart(ctxSD, {
                type: 'line',
                data: {
                    labels: data.supply_demand_match.index,
                    datasets: [
                        {
                            label: 'Supply Level (Stock)',
                            data: data.supply_demand_match.supply,
                            borderColor: '#fbbf24',
                            borderWidth: 2,
                            fill: false,
                            pointRadius: 2,
                            tension: 0.3
                        },
                        {
                            label: 'Market Demand',
                            data: data.supply_demand_match.demand,
                            borderColor: '#3b82f6',
                            borderWidth: 2,
                            fill: false,
                            pointRadius: 2,
                            tension: 0.3
                        },
                        {
                            label: 'Actual Sold',
                            data: data.supply_demand_match.sold,
                            borderColor: '#10b981',
                            borderWidth: 2.5,
                            fill: false,
                            pointRadius: 3,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } } }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#94a3b8', maxTicksLimit: 15 } },
                        y: { 
                            grid: { color: 'rgba(255,255,255,0.02)' }, 
                            ticks: { 
                                color: '#94a3b8',
                                stepSize: 1,
                                callback: function(val) {
                                    if (val === 1) return "Low";
                                    if (val === 2) return "Med";
                                    if (val === 3) return "High";
                                    return "";
                                }
                            } 
                        }
                    }
                }
            });

            // CHART 3: Action Profile (Doughnut Chart)
            const ctxAct = document.getElementById("actionDistChart").getContext("2d");
            const actLabels = Object.keys(data.action_distribution);
            const actValues = Object.values(data.action_distribution);
            new Chart(ctxAct, {
                type: 'doughnut',
                data: {
                    labels: actLabels,
                    datasets: [{
                        data: actValues,
                        backgroundColor: ['#10b981', '#fbbf24', '#3b82f6', '#f97316'],
                        borderWidth: 1,
                        borderColor: '#1e293b'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            position: 'bottom',
                            labels: { color: '#94a3b8', boxWidth: 10, padding: 8, font: { family: 'Outfit', size: 8.5 } } 
                        }
                    }
                }
            });

            // CHART 4: Spoilage by Category (Horizontal Bar Chart)
            const ctxWaste = document.getElementById("categoryWasteChart").getContext("2d");
            new Chart(ctxWaste, {
                type: 'bar',
                data: {
                    labels: data.waste_by_category.categories.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
                    datasets: [
                        {
                            label: 'Q-Learning Waste',
                            data: data.waste_by_category.q_learning,
                            backgroundColor: '#10b981',
                            borderColor: '#059669',
                            borderWidth: 1
                        },
                        {
                            label: 'Baseline Waste',
                            data: data.waste_by_category.baseline,
                            backgroundColor: '#ef4444',
                            borderColor: '#dc2626',
                            borderWidth: 1
                        }
                    ]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { 
                            position: 'bottom',
                            labels: { color: '#94a3b8', boxWidth: 10, padding: 8, font: { family: 'Outfit', size: 8.5 } } 
                        }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255,255,255,0.02)' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'none' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        })
        .catch(err => console.error("Error evaluating predictions:", err));

    // ------------------------------------------------
    // 3. Fetch Q-Table & Populate Interactive Heatmap
    // ------------------------------------------------
    fetch("/api/q_table")
        .then(res => res.json())
        .then(qTable => {
            globalQTable = qTable;
            buildHeatmap();
            updateSimulation(); // Run initial simulation update
        })
        .catch(err => {
            console.error("Error loading Q-table:", err);
            heatmapGrid.innerHTML = "<p style='grid-column: span 3; text-align:center;'>Error loading Q-table data</p>";
        });

    function buildHeatmap() {
        const stockLevels = ["High", "Medium", "Low"];
        const demandLevels = ["Low", "Medium", "High"];
        
        heatmapGrid.innerHTML = "";
        
        stockLevels.forEach(stock => {
            demandLevels.forEach(demand => {
                const key = `China|cattle|${stock}|${demand}`;
                let optAction = 0;
                
                if (globalQTable && globalQTable[key]) {
                    const qVals = globalQTable[key];
                    optAction = qVals.indexOf(Math.max(...qVals));
                } else {
                    // Fallbacks matching general guidelines
                    if (stock === "High" && demand === "Low") optAction = 1;
                    else if (stock === "High" && demand === "Medium") optAction = 3;
                    else if (stock === "Medium" && demand === "Low") optAction = 2;
                    else optAction = 0;
                }
                
                const cell = document.createElement("div");
                cell.className = `heatmap-cell action-${optAction}`;
                cell.dataset.supply = stock;
                cell.dataset.demand = demand;
                
                const stateLabel = document.createElement("span");
                stateLabel.className = "state-lbl";
                stateLabel.innerText = `S:${stock[0]} | D:${demand[0]}`;
                
                const actionLabel = document.createElement("span");
                actionLabel.className = "action-lbl";
                actionLabel.innerText = optAction === 0 ? "Standard" : (optAction === 1 ? "Discount" : (optAction === 2 ? "Refrigerate" : "Repurpose"));
                
                cell.appendChild(stateLabel);
                cell.appendChild(actionLabel);
                
                // Mouse interactions
                cell.addEventListener("mouseenter", () => {
                    heatmapEx.innerHTML = `<strong>State:</strong> Supply <strong>${stock}</strong>, Demand <strong>${demand}</strong>.<br/>` +
                                        `<strong>Policy Action:</strong> <span>${actionsDesc[optAction]}</span> — ${actionsExplain[optAction]}`;
                });
                
                // Click interaction: feeds states into the simulator
                cell.addEventListener("click", () => {
                    // Highlight selected cell
                    document.querySelectorAll(".heatmap-cell").forEach(c => c.classList.remove("active-cell"));
                    cell.classList.add("active-cell");
                    
                    // Set dropdown values
                    simSupply.value = stock;
                    simDemand.value = demand;
                    
                    // Trigger simulation update
                    updateSimulation();
                    
                    heatmapEx.innerHTML = `<strong>Flipped Simulator State:</strong> supply chain loaded with <strong>${stock} Stock</strong> and <strong>${demand} Demand</strong>.<br/>` +
                                        `<em>Check the Node Network layout above to see color updates.</em>`;
                });
                
                heatmapGrid.appendChild(cell);
            });
        });
    }

    // ------------------------------------------------
    // 4. Interactive Simulator Logic
    // ------------------------------------------------
    function updateSimulation() {
        const cat = simCategory.value;
        const supply = simSupply.value;
        const demand = simDemand.value;
        
        // Find Q-learning recommended action for selected state
        const stateKey = `China|${cat}|${supply}|${demand}`;
        let action = 0;
        
        if (globalQTable && globalQTable[stateKey]) {
            const qVals = globalQTable[stateKey];
            action = qVals.indexOf(Math.max(...qVals));
        } else {
            // General heuristics
            if (supply === "High" && demand === "Low") action = 1;
            else if (supply === "High" && demand === "Medium") action = 3;
            else if (supply === "Medium" && demand === "Low") action = 2;
            else action = 0;
        }
        
        // Update nodes details based on action and selected states
        const supplyVal = supply === "Low" ? 1 : (supply === "Medium" ? 2 : 3);
        const demandVal = demand === "Low" ? 1 : (demand === "Medium" ? 2 : 3);
        
        // Simulating node status
        const nodeStatus = {
            "Producer": "green",
            "Warehouse": "green",
            "Distributor": "green",
            "Retailer": "green"
        };
        
        const nodeStocks = {
            "Producer": supplyVal * 35,
            "Warehouse": supplyVal * 35,
            "Distributor": supplyVal * 30,
            "Retailer": supplyVal * 25
        };
        
        // 1. Warehouse status - refrigeration protects surplus, standard decays
        if (supply === "High" && action !== 2 && action !== 3) {
            nodeStatus["Warehouse"] = "red"; // High waste hazard
        } else if (supply === "High" && (action === 2 || action === 3)) {
            nodeStatus["Warehouse"] = "yellow"; // Stabilized
        }
        
        // 2. Retailer status - stock matches demand
        if (supplyVal === demandVal) {
            nodeStatus["Retailer"] = "green";
        } else if (supplyVal > demandVal) {
            // Excess inventory
            if (action === 1) { // Discount stimulates demand, making it yellow (moderate)
                nodeStatus["Retailer"] = "yellow";
                nodeStocks["Retailer"] = (supplyVal - 0.5) * 25;
            } else {
                nodeStatus["Retailer"] = "red"; // Unsold inventory waste
            }
        } else {
            // Supply shortage
            nodeStatus["Retailer"] = "yellow"; // shortage bottleneck
        }
        
        // Update node elements in SVG
        const nodes = ["Producer", "Warehouse", "Distributor", "Retailer"];
        nodes.forEach(node => {
            const group = document.getElementById(`node-${node}`);
            if (group) {
                // Class mapping
                group.setAttribute("class", `node-group node-${nodeStatus[node]}`);
                
                // Update text elements
                const actionTxt = group.querySelector(".node-action");
                const infoTxt = group.querySelector(".node-info");
                
                // Producer shows Standard. Warehouse/Retailer show Q-learning action recommendations
                if (node === "Producer" || node === "Distributor") {
                    if (actionTxt) actionTxt.textContent = "Standard";
                } else if (node === "Warehouse") {
                    if (actionTxt) actionTxt.textContent = (action === 2) ? "Refrigerating" : ((action === 3) ? "Repurposing" : "Standard");
                } else {
                    if (actionTxt) actionTxt.textContent = (action === 1) ? "Discounting" : "Standard";
                }
                
                if (infoTxt) infoTxt.textContent = `Stock: ${supply}`;
                
                // Adjust core size slightly based on inventory scale
                const core = group.querySelector(".node-core");
                if (core) {
                    const r = 30 + (supplyVal * 5);
                    core.setAttribute("r", r);
                }
                const glow = group.querySelector(".node-glow");
                if (glow) {
                    const r = 40 + (supplyVal * 5);
                    glow.setAttribute("r", r);
                }
            }
        });
        
        // Adjust connection link active flow animations
        let activeLinks = true;
        let flowSpeed = "1.5s";
        
        // If there's a red bottleneck, slow down flows
        const statusVals = Object.values(nodeStatus);
        if (statusVals.includes("red")) {
            flowSpeed = "6s";
        } else if (statusVals.includes("yellow")) {
            flowSpeed = "3.2s";
        }
        
        [link1, link2, link3].forEach(link => {
            if (activeLinks) {
                link.setAttribute("class", "flow-line active");
                link.style.animationDuration = flowSpeed;
            } else {
                link.setAttribute("class", "flow-line");
            }
        });
    }

    // Connect control events
    simCategory.addEventListener("change", updateSimulation);
    simSupply.addEventListener("change", updateSimulation);
    simDemand.addEventListener("change", updateSimulation);
});
