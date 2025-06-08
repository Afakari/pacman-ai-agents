const fs = require('fs').promises;

async function mergeJSONs() {
    try {
        // Read the two JSON files
        const data1 = JSON.parse(await fs.readFile('metrics_data.json', 'utf8'));
        const data2 = JSON.parse(await fs.readFile('new_metrics.json', 'utf8'));

        // Merge labels
        const mergedLabels = [...data1.labels, ...data2.labels];

        // Merge metrics
        const mergedMetrics = {
            a_star: {
                execution_time: [...data1.metrics.a_star.execution_time, ...data2.metrics.a_star.execution_time],
                memory_usage: [...data1.metrics.a_star.memory_usage, ...data2.metrics.a_star.memory_usage],
                nodes_expanded: [...data1.metrics.a_star.nodes_expanded, ...data2.metrics.a_star.nodes_expanded],
                path_length: [...data1.metrics.a_star.path_length, ...data2.metrics.a_star.path_length]
            },
            rbfs: {
                execution_time: [...data1.metrics.rbfs.execution_time, ...data2.metrics.rbfs.execution_time],
                memory_usage: [...data1.metrics.rbfs.memory_usage, ...data2.metrics.rbfs.memory_usage],
                nodes_expanded: [...data1.metrics.rbfs.nodes_expanded, ...data2.metrics.rbfs.nodes_expanded],
                path_length: [...data1.metrics.rbfs.path_length, ...data2.metrics.rbfs.path_length]
            }
        };

        // Create merged JSON object
        const mergedJSON = {
            labels: mergedLabels,
            metrics: mergedMetrics
        };

        // Write merged JSON to a new file
        await fs.writeFile('merged_metrics.json', JSON.stringify(mergedJSON, null, 2));
        console.log('Merged JSON saved to merged_metrics.json');
    } catch (error) {
        console.error('Error merging JSONs:', error);
    }
}

mergeJSONs();
