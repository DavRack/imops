use std::env;
use pichromatic_pipeline::modules::get_pipeline_schema;

fn main() {
    let args: Vec<String> = env::args().collect();
    let print_json = args.contains(&"--json".to_string());

    let schemas = get_pipeline_schema();

    if print_json {
        let json_str = serde_json::to_string_pretty(&schemas).unwrap();
        println!("{}", json_str);
    } else {
        println!("====================================================");
        println!("  Pichromatic Pipeline Modules Specification Schema  ");
        println!("====================================================\n");
        println!("This specification lists all available image processing modules,");
        println!("their properties, default values, and parameter types.\n");
        println!("Tip: Run with --json for raw machine-readable output.\n");

        for module in schemas {
            println!("Module: {}", module.name);
            println!("Description: {}", module.description);
            if module.fields.is_empty() {
                println!("  (No configuration parameters required)\n");
            } else {
                println!("  Parameters:");
                for field in module.fields {
                    let mut type_info = field.field_type.to_string();
                    if let Some(ref choices) = field.choices {
                        type_info = format!("{} (choices: {:?})", type_info, choices);
                    }
                    println!("    - Name:        {}", field.name);
                    println!("      Type:        {}", type_info);
                    println!("      Default:     {}", field.default_value);
                    if !field.description.is_empty() {
                        println!("      Description: {}", field.description);
                    }
                    println!();
                }
            }
            println!("----------------------------------------------------\n");
        }
    }
}
